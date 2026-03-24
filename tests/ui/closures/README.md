This directory contains general closure tests that do not fit a more specific
closure subdirectory.

Tests here usually focus on:

- capture behavior
- closure kind (`Fn`, `FnMut`, `FnOnce`)
- coercions involving closures or closure trait objects
- general closure diagnostics

More specialized closure topics live in subdirectories such as:

- `2229_closure_analysis/` for RFC 2229 capture analysis tests
- `closure-expected-type/` for expected-type-driven closure argument inference
- `deduce-signature/` for closure signature deduction from bounds and
  obligations
- `print/` for closure printing tests
