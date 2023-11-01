The tests in this directory are shared by two different test modes, and can be
run in multiple different ways:

- `./x.py test coverage-map` (compiles to LLVM IR and checks coverage mappings)
- `./x.py test run-coverage` (runs a test binary and checks its coverage report)
- `./x.py test coverage` (runs both `coverage-map` and `run-coverage`)

## Maintenance note

These tests can be sensitive to small changes in MIR spans or MIR control flow,
especially in HIR-to-MIR lowering or MIR optimizations.

If you haven't touched the coverage code directly, and the tests still pass in
`run-coverage` mode, then it should usually be OK to just re-bless the mappings
as necessary with `./x.py test coverage-map --bless`, without worrying too much
about the exact changes.
