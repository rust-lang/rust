# `print=check-cfg`

The tracking issue for this feature is: [#125704](https://github.com/rust-lang/rust/issues/125704).

------------------------

This option of the `--print` flag print the list of all the expected cfgs.

This is related to the [`--check-cfg` flag][check-cfg] which allows specifying arbitrary expected
names and values.

This print option works similarly to `--print=cfg` (modulo check-cfg specifics).

| `--check-cfg`                     | `--print=check-cfg`         |
|-----------------------------------|-----------------------------|
| `cfg(foo)`                        | `foo`                       |
| `cfg(foo, values("bar"))`         | `foo="bar"`                 |
| `cfg(foo, values(none(), "bar"))` | `foo` & `foo="bar"`         |
|                                   | *check-cfg specific syntax* |
| `cfg(foo, values(any())`          | `foo=any()`                 |
| `cfg(foo, values())`              | `foo=`                      |
| `cfg(any())`                      | `any()`                     |
| *none*                            | `any()=any()`               |

To be used like this:

```bash
rustc --print=check-cfg -Zunstable-options lib.rs
```

[check-cfg]: https://doc.rust-lang.org/nightly/rustc/check-cfg.html
