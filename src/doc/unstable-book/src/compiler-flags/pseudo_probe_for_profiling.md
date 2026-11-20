# `pseudo-probe-for-profiling`

---

`-Zpseudo-probe-for-profiling` inserts pseudo probes into each function, same as
Clang's `-fpseudo-probe-for-profiling` does. Pseudo probes are required for
context-sensitive PGO (CSSPGO). Probes need debug info to record call sites and
the inline context, use together with `-Zdebuginfo-for-profiling`, or at least
`-Cdebuginfo=line-tables-only`.

This flag cannot be used with `-Cprofile-generate` or `-Cprofile-use`.
