# `print=check-cfg`

The tracking issue for this feature is: [#125704](https://github.com/rust-lang/rust/issues/125704).

------------------------

This option of the `--print` flag print the list of expected cfgs.

This is related to the `--check-cfg` flag which allows specifying arbitrary expected
names and values.

This print option works similarly to `--print=cfg` (modulo check-cfg specifics):
 - *check_cfg syntax*: *output of --print=check-cfg*
 - `cfg(windows)`: `windows`
 - `cfg(feature, values("foo", "bar"))`: `feature="foo"` and `feature="bar"`
 - `cfg(feature, values(none(), ""))`: `feature` and `feature=""`
 - `cfg(feature, values(any()))`: `feature=any()`
 - `cfg(feature, values())`: `feature=`
 - `cfg(any())`: `any()`
 - *nothing*: `any()=any()`

To be used like this:

```bash
rustc --print=check-cfg -Zunstable-options lib.rs
```
