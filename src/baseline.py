"""Keyword-rule baseline. Bilingual (EN/RU), stdlib only, $0 per 1000 calls.

This is a genuine attempt, not a strawman. If an LLM cannot clear this bar by
a margin worth its latency and cost, the honest recommendation is to ship this
instead. Rules are ordered: the first matching group wins, most specific first.
"""
import re

# Ordered most-specific-first. Within a tier, any hit claims the row.
RULES = [
    ("goal_met", [
        r"\bpassed\b", r"\bgot the job\b", r"\bjob offer\b", r"\bgraduat",
        r"\bcertif\w* (?:last|this)\b", r"\bfinished (?:the )?(?:full|all|my)\b",
        r"\bcompleted all\b", r"\bmission accomplished\b", r"\bthesis is submitted\b",
        r"\blearned what i came for\b", r"\bswitched careers\b", r"\bno longer need\b",
        r"сдал экзамен", r"цель достигнута", r"повысили на работе",
    ]),
    ("technical", [
        r"\bcrash", r"\bbuffer", r"\bbug\b", r"\b500 error\b", r"\bbroken\b",
        r"\bcannot log in\b", r"\bprogress got reset\b", r"\btimes? out\b",
        r"\bdisappear", r"\bextremely slow\b", r"\bstopped working\b",
        r"\bmarked wrong\b", r"\bwent through twice\b",
        r"не загружается", r"вылетает", r"не работает",
    ]),
    ("support", [
        r"\bticket\b", r"\bnobody (?:has )?replied\b", r"\bno(?:body)? answer",
        r"\brude\b", r"\bbot reply\b", r"\brefund\b", r"\bsurprise \w+ charge\b",
        r"\breach a human\b", r"\bhelp center\b", r"\bcallback\b",
        r"\bthree agents\b", r"\b\d+ emails\b", r"\bcancel and resubscribe\b",
        r"\bhow the last conversation went\b",
        r"поддержка не отвечает", r"вернуть деньги", r"отказали",
    ]),
    ("competitor", [
        r"\bcoursera\b", r"\budemy\b", r"\bdatacamp\b", r"\bstepik\b",
        r"\bbootcamp\b", r"\banother (?:provider|platform)\b", r"\bcompetitor\b",
        r"\bswitching to\b", r"\bmoving to\b", r"\banother lms\b",
        r"\bjob guarantee\b", r"\bmasters program\b", r"\benterprise deal\b",
        r"\b1-on-1 code review\b", r"\blive mentorship\b",
        r"перехожу на", r"живыми вебинарами",
    ]),
    ("time", [
        r"\bno time\b", r"\btoo busy\b", r"\btaking a break\b", r"\bno capacity\b",
        r"\bhave not opened\b", r"\bkeep not watching\b", r"\bpause\b", r"\bfreeze\b",
        r"\bcome back (?:in|later)\b", r"\bstep away\b", r"\bbaby arrived\b",
        r"\bfinals\b", r"\bschedule\b", r"\bnot happening this year\b",
        r"нет времени", r"не успеваю", r"заморозить", r"разгруз",
    ]),
    ("price", [
        r"\btoo expensive\b", r"\bexpensive\b", r"\btoo pric", r"\bcannot afford\b",
        r"\bbudget\b", r"\blost my job\b", r"\bnot worth (?:it|the money)\b",
        r"\bcost\b", r"\bprice point\b", r"\bthird of the price\b",
        r"\bcheaper\b", r"\bjustify the cost\b", r"\breimburs",
        r"дорого", r"не по карману", r"большие деньги", r"скидк",
    ]),
    ("content_gap", [
        r"\bbeginner level\b", r"\bno courses on\b", r"\boutdated\b",
        r"\breads slides\b", r"\blearned nothing\b", r"\btoo much theory\b",
        r"\bno practice\b", r"\bzero practice\b", r"\bcontent stops\b",
        r"\bsubtitles\b", r"\bno math prerequisites\b", r"\bgeneric python\b",
        r"\bnothing left to watch\b", r"\bevery track is an intro\b",
        r"нет материалов", r"пересказ учебника", r"нет курсов",
    ]),
]

# Texts this short or this devoid of content are abstentions by construction.
LOW_SIGNAL = {
    "cancel", "-", "no comment", "ok", "asdfgh", "не хочу больше",
}


def classify(text):
    t = (text or "").strip()
    low = t.lower()

    if low in LOW_SIGNAL or len(low) < 4:
        return "unclear"

    for label, patterns in RULES:
        for pat in patterns:
            if re.search(pat, low):
                return label
    return "unclear"
