# `force-full-debuginfo`

The tracking issue for this feature is: [#64405](https://github.com/rust-lang/rust/issues/64405).

---

The option `-Z force-full-debuginfo` controls whether `-C debuginfo=1` generates full debug info for
a codegen-unit.  Due to an oversight, debuginfo=1 (which should only mean "line tables") generated
additional debuginfo for many years.  Due to backwards compatibility concerns, we are not yet
changing that meaning, but instead adding this flag to allow opting-in to the new, reduced, debuginfo.

Supported options for this value are:
- `yes` - the default, include full debuginfo for the codegen unit
- `no`  - include only line info for the codegen unit

The default for this option may change in the future, but it is unlikely to be stabilized.
