# Contributing

Paper suggestions, corrections, code links, and clearer summaries are welcome. Open an issue or pull request with a stable paper URL and a sentence about why it belongs here.

## Suggest a paper or resource

For a paper, include its full title, arXiv ID, DOI, or official venue URL, and a short explanation of the contribution. Tell us what learns from reward: the generator, an action policy, a critic, or a separate controller. Note its domain, method family, and online/offline setting where known. An abstract is enough to suggest a candidate; method-level evidence supports a reading highlight.

For code, prefer a repository linked by the authors or their project. For a survey, tutorial, or toolkit, explain who it helps and what it covers. Highlights reflect explanatory value or technical promise, independent of citation counts.

Keep one full entry per paper and use anchor cross-links for other categories. Include earlier titles as aliases of the same stable identity; shared acronyms can refer to different papers. Please flag identity conflicts, unavailable code, or a summary that confuses generator training with inference steering or distillation.

## Edit the list

Keep the public list concise and readable:

- Put a paper in Core Methods when its primary contribution is a reusable learning algorithm best understood by method family, even if its experiments focus on one domain. Keep methods whose main contribution depends on domain-specific architectures, rewards, representations, environment structure, or scientific objects under their domain. Use a stable paper link and author-linked code or project page when available.
- Give each Core Methods entry exactly one subtle `Subdomain` badge for the domains actually developed or evaluated in the original paper, not presumed applicability. Use only `Images & Video`, `World Models`, `Language & Reasoning`, `Robotics & Control`, `Scientific Discovery`, and `Audio`; combine multiple values with ` · ` in one badge. Reuse the existing light badge style and colors for the same value. Domain-specific entries normally need no badge; do not add method or training-regime badges.
- Write one sentence explaining the algorithmic contribution rather than repeating the abstract.
- Place model and system reports in Tech Reports, with a Subdomain badge and a concise description of their reward-training recipe. Keep separately published learning algorithms in Core Methods and cross-link related reports.
- Use one full entry per paper and anchor links for secondary categories.
- Reserve ⭐ Fundamental for work that explains the field and 🚀 Frontier & Potential for especially promising current directions. These markers express editorial judgment, not ordering or citation rank.
- Keep benchmarks, reward models, inference steering, and distillation clearly distinguished from generator or policy training.

Small corrections can be submitted directly as pull requests. For larger additions, opening an issue first helps avoid duplicate work.

For badges, use Shields SVGs with `style=flat` and `labelColor=e8ecef`. Keep these light value colors fixed:

| Subdomain | Value color |
|---|---|
| Images & Video | `dce8f3` |
| World Models | `ddebe8` |
| Language & Reasoning | `e5e0f1` |
| Robotics & Control | `e1ead8` |
| Scientific Discovery | `f0e6d5` |
| Audio | `efdee6` |

For a combined footprint, keep the order above and use the rounded mean of the component RGB colors in one badge. URL-encode the value, including `&` and ` · `.
