# codegen-units/partitioning tests

This test suite is designed to test that codegen unit partitioning works as intended.
Note that it does not evaluate whether CGU partitioning is *good*. That is the job of the compiler benchmark suite.

All tests in this suite use the flag `-Zprint-mono-items`, which makes the compiler print a machine-readable summary of all MonoItems that were collected, which CGUs they were assigned to, and the linkage in each CGU. The output looks like:
```
MONO_ITEM <item> @@ <cgu name>[<linkage>] <other cgu name>[<linkage in other cgu>]
```

The current CGU partitioning algorithm essentially groups MonoItems by which module they are defined in, then merges small CGUs. There are a lot of inline modules in this test suite because that's the only way to observe the partitioning.

Currently, the test suite is very heavily biased towards incremental builds with -Copt-level=0. This is mostly an accident of history; the entire test suite was added as part of supporting incremental compilation in #32779. But also CGU partitioning is *mostly* valuable because the CGU is the unit of incrementality to the codegen backend (cached queries are the unit of incrementality for the rest of the compiler).
