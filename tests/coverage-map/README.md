The tests in `./status-quo` were copied from `tests/run-coverage` in order to
capture the current behavior of the instrumentor on non-trivial programs.
The actual mappings have not been closely inspected.

## Maintenance note

These tests can be sensitive to small changes in MIR spans or MIR control flow,
especially in HIR-to-MIR lowering or MIR optimizations.

If you haven't touched the coverage code directly, and the `run-coverage` test
suite still works, then it should usually be OK to just `--bless` these
coverage mapping tests as necessary, without worrying too much about the exact
changes.
