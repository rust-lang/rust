# `print=check-cfg`

The tracking issue for this feature is: [#125704](https://github.com/rust-lang/rust/issues/125704).

------------------------

This option of the `--print` flag print the list of all the expected cfgs.

This is related to the [`--check-cfg` flag][check-cfg] which allows specifying arbitrary expected
names and values.

This print option outputs compatible `--check-cfg` arguments with a reduced syntax where all the
expected values are on the same line and `values(...)` is always explicit.

| `--check-cfg`                     | `--print=check-cfg`               |
|-----------------------------------|-----------------------------------|
| `cfg(foo)`                        | `cfg(foo, values(none()))         |
| `cfg(foo, values("bar"))`         | `cfg(foo, values("bar"))`         |
| `cfg(foo, values(none(), "bar"))` | `cfg(foo, values(none(), "bar"))` |
| `cfg(foo, values(any())`          | `cfg(foo, values(any())`          |
| `cfg(foo, values())`              | `cfg(foo, values())`              |
| `cfg(any())`                      | `cfg(any())`                      |
| *nothing*                         | *nothing*                         |

The print option includes well known cfgs.

To be used like this:

```bash
rustc --print=check-cfg -Zunstable-options lib.rs
```

> **Note:** Users should be resilient when parsing, in particular against new predicates that
may be added in the future.

[check-cfg]: https://doc.rust-lang.org/nightly/rustc/check-cfg.html
