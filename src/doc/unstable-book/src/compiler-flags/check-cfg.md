# `check-cfg`

The tracking issue for this feature is: [#82450](https://github.com/rust-lang/rust/issues/82450).

------------------------

This feature enables checking of conditional configuration.

`rustc` accepts the `--check-cfg` option, which specifies whether to check conditions and how to
check them. The `--check-cfg` option takes a value, called the _check cfg specification_.
This specification has one form:

1. `--check-cfg cfg(...)` mark a configuration and it's expected values as expected.

*No implicit expectation is added when using `--cfg`. Users are expected to
pass all expected names and values using the _check cfg specification_.*

## The `cfg(...)` form

The `cfg(...)` form enables checking the values within list-valued conditions. It has this
basic form:

```bash
rustc --check-cfg 'cfg(name, values("value1", "value2", ... "valueN"))'
```

where `name` is a bare identifier (has no quotes) and each `"value"` term is a quoted literal
string. `name` specifies the name of the condition, such as `feature` or `my_cfg`.

When the `cfg(...)` option is specified, `rustc` will check every `#[cfg(name = "value")]`
attribute, `#[cfg_attr(name = "value")]` attribute, `#[link(name = "a", cfg(name = "value"))]`
attribute and `cfg!(name = "value")` macro call. It will check that the `"value"` specified is
present in the list of expected values. If `"value"` is not in it, then `rustc` will report an
`unexpected_cfgs` lint diagnostic. The default diagnostic level for this lint is `Warn`.

*The command line `--cfg` arguments are currently *NOT* checked but may very well be checked in
the future.*

To check for the _none_ value (ie `#[cfg(foo)]`) one can use the `none()` predicate inside
`values()`: `values(none())`. It can be followed or precessed by any number of `"value"`.

To enable checking of values, but to provide an *none*/empty set of expected values
(ie. expect `#[cfg(name)]`), use these forms:

```bash
rustc --check-cfg 'cfg(name)'
rustc --check-cfg 'cfg(name, values(none()))'
```

To enable checking of name but not values, use one of these forms:

  - No expected values (_will lint on every value_):
    ```bash
    rustc --check-cfg 'cfg(name, values())'
    ```

  - Unknown expected values (_will never lint_):
    ```bash
    rustc --check-cfg 'cfg(name, values(any()))'
    ```

To avoid repeating the same set of values, use this form:

```bash
rustc --check-cfg 'cfg(name1, ..., nameN, values("value1", "value2", ... "valueN"))'
```

The `--check-cfg cfg(...)` option can be repeated, both for the same condition name and for
different names. If it is repeated for the same condition name, then the sets of values for that
condition are merged together (precedence is given to `values(any())`).

## Well known names and values

`rustc` has a internal list of well known names and their corresponding values.
Those well known names and values follows the same stability as what they refer to.

Well known names and values checking is always enabled as long as at least one
`--check-cfg` argument is present.

As of `2024-02-15T`, the list of known names is as follows:

<!--- See CheckCfg::fill_well_known in compiler/rustc_session/src/config.rs -->

 - `clippy`
 - `debug_assertions`
 - `doc`
 - `doctest`
 - `miri`
 - `overflow_checks`
 - `panic`
 - `proc_macro`
 - `relocation_model`
 - `sanitize`
 - `sanitizer_cfi_generalize_pointers`
 - `sanitizer_cfi_normalize_integers`
 - `target_abi`
 - `target_arch`
 - `target_endian`
 - `target_env`
 - `target_family`
 - `target_feature`
 - `target_has_atomic`
 - `target_has_atomic_equal_alignment`
 - `target_has_atomic_load_store`
 - `target_os`
 - `target_pointer_width`
 - `target_thread_local`
 - `target_vendor`
 - `test`
 - `unix`
 - `windows`

Like with `values(any())`, well known names checking can be disabled by passing `cfg(any())`
as argument to `--check-cfg`.

## Examples

### Equivalence table

This table describe the equivalence of a `--cfg` argument to a `--check-cfg` argument.

| `--cfg`                       | `--check-cfg`                                              |
|-------------------------------|------------------------------------------------------------|
| *nothing*                     | *nothing* or `--check-cfg=cfg()` (to enable the checking)  |
| `--cfg foo`                   | `--check-cfg=cfg(foo)` or `--check-cfg=cfg(foo, values(none()))` |
| `--cfg foo=""`                | `--check-cfg=cfg(foo, values(""))`                         |
| `--cfg foo="bar"`             | `--check-cfg=cfg(foo, values("bar"))`                      |
| `--cfg foo="1" --cfg foo="2"` | `--check-cfg=cfg(foo, values("1", "2"))`                   |
| `--cfg foo="1" --cfg bar="2"` | `--check-cfg=cfg(foo, values("1")) --check-cfg=cfg(bar, values("2"))` |
| `--cfg foo --cfg foo="bar"`   | `--check-cfg=cfg(foo, values(none(), "bar"))`              |

### Example: Cargo-like `feature` example

Consider this command line:

```bash
rustc --check-cfg 'cfg(feature, values("lion", "zebra"))' \
      --cfg 'feature="lion"' -Z unstable-options example.rs
```

This command line indicates that this crate has two features: `lion` and `zebra`. The `lion`
feature is enabled, while the `zebra` feature is disabled.
Given the `--check-cfg` arguments, exhaustive checking of names and
values are enabled.

`example.rs`:
```rust
#[cfg(feature = "lion")]     // This condition is expected, as "lion" is an expected value of `feature`
fn tame_lion(lion: Lion) {}

#[cfg(feature = "zebra")]    // This condition is expected, as "zebra" is an expected value of `feature`
                             // but the condition will still evaluate to false
                             // since only --cfg feature="lion" was passed
fn ride_zebra(z: Zebra) {}

#[cfg(feature = "platypus")] // This condition is UNEXPECTED, as "platypus" is NOT an expected value of
                             // `feature` and will cause a compiler warning (by default).
fn poke_platypus() {}

#[cfg(feechure = "lion")]    // This condition is UNEXPECTED, as 'feechure' is NOT a expected condition
                             // name, no `cfg(feechure, ...)` was passed in `--check-cfg`
fn tame_lion() {}

#[cfg(windows = "unix")]     // This condition is UNEXPECTED, as while 'windows' is a well known
                             // condition name, it doens't expect any values
fn tame_windows() {}
```

### Example: Multiple names and values

```bash
rustc --check-cfg 'cfg(is_embedded, has_feathers)' \
      --check-cfg 'cfg(feature, values("zapping", "lasers"))' \
      --cfg has_feathers --cfg 'feature="zapping"' -Z unstable-options
```

```rust
#[cfg(is_embedded)]         // This condition is expected, as 'is_embedded' was provided in --check-cfg
fn do_embedded() {}         // and doesn't take any value

#[cfg(has_feathers)]        // This condition is expected, as 'has_feathers' was provided in --check-cfg
fn do_features() {}         // and doesn't take any value

#[cfg(has_mumble_frotz)]    // This condition is UNEXPECTED, as 'has_mumble_frotz' was NEVER provided
                            // in any --check-cfg arguments
fn do_mumble_frotz() {}

#[cfg(feature = "lasers")]  // This condition is expected, as "lasers" is an expected value of `feature`
fn shoot_lasers() {}

#[cfg(feature = "monkeys")] // This condition is UNEXPECTED, as "monkeys" is NOT an expected value of
                            // `feature`
fn write_shakespeare() {}
```

### Example: Condition names without values

```bash
rustc --check-cfg 'cfg(is_embedded, has_feathers, values(any()))' \
      --cfg has_feathers -Z unstable-options
```

```rust
#[cfg(is_embedded)]      // This condition is expected, as 'is_embedded' was provided in --check-cfg
                         // as condition name
fn do_embedded() {}

#[cfg(has_feathers)]     // This condition is expected, as "has_feathers" was provided in --check-cfg
                         // as condition name
fn do_features() {}

#[cfg(has_feathers = "zapping")] // This condition is expected, as "has_feathers" was provided in
                                 // and because *any* values is expected for 'has_feathers' no
                                 // warning is emitted for the value "zapping"
fn do_zapping() {}

#[cfg(has_mumble_frotz)] // This condition is UNEXPECTED, as 'has_mumble_frotz' was not provided
                         // in any --check-cfg arguments
fn do_mumble_frotz() {}
```
