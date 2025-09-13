# Checking conditional configurations

`rustc` supports checking that every _reachable_[^reachable] `#[cfg]` matches a list of the
expected config names and values.

This can help with verifying that the crate is correctly handling conditional compilation for
different target platforms or features. It ensures that the cfg settings are consistent between
what is intended and what is used, helping to catch potential bugs or errors early in the
development process.

In order to accomplish that goal, `rustc` accepts the `--check-cfg` flag, which specifies
whether to check conditions and how to check them.

> **Note:** For interacting with this through Cargo,
see [Cargo Specifics](check-cfg/cargo-specifics.md) page.

[^reachable]: `rustc` promises to at least check reachable `#[cfg]`, and while non-reachable
`#[cfg]` are not currently checked, they may well be checked in the future without it being a
breaking change.

## Specifying expected names and values

To specify expected names and values, the _check cfg specification_ provides the `cfg(...)`
option which enables specifying for an expected config name and it's expected values.

> **Note:** No implicit expectation is added when using `--cfg`. Users are expected to
pass all expected names and values using the _check cfg specification_.

It has this basic form:

```bash
rustc --check-cfg 'cfg(name, values("value1", "value2", ... "valueN"))'
```

where `name` is a bare identifier (has no quotes) and each `"value"` term is a quoted literal
string. `name` specifies the name of the condition, such as `feature` or `my_cfg`.
`"value"` specify one of the value of that condition name.

When the `cfg(...)` option is specified, `rustc` will check every[^reachable]:
 - `#[cfg(name = "value")]` attribute
 - `#[cfg_attr(name = "value")]` attribute
 - `#[link(name = "a", cfg(name = "value"))]` attribute
 -  `cfg!(name = "value")` macro call

> *The command line `--cfg` arguments are currently NOT checked but may very well be checked
in the future.*

`rustc` will check that the `"value"` specified is present in the list of expected values.
If `"value"` is not in it, then `rustc` will report an `unexpected_cfgs` lint diagnostic.
The default diagnostic level for this lint is `Warn`.

To check for the _none_ value (ie `#[cfg(foo)]`) one can use the `none()` predicate inside
`values()`: `values(none())`. It can be followed or preceded by any number of `"value"`.

To enable checking of values, but to provide an *none*/empty set of expected values
(ie. expect `#[cfg(name)]`), use these forms:

```bash
rustc --check-cfg 'cfg(name)'
rustc --check-cfg 'cfg(name, values(none()))'
```

To enable checking of name but not values, use one of these forms:

  - No expected values (_will lint on every value of `name`_):
    ```bash
    rustc --check-cfg 'cfg(name, values())'
    ```

  - Unknown expected values (_will never lint on value of `name`_):
    ```bash
    rustc --check-cfg 'cfg(name, values(any()))'
    ```

To avoid repeating the same set of values, use this form:

```bash
rustc --check-cfg 'cfg(name1, ..., nameN, values("value1", "value2", ... "valueN"))'
```

To enable checking without specifying any names or values, use this form:

```bash
rustc --check-cfg 'cfg()'
```

The `--check-cfg cfg(...)` option can be repeated, both for the same condition name and for
different names. If it is repeated for the same condition name, then the sets of values for that
condition are merged together (precedence is given to `values(any())`).

> To help out an equivalence table between `--cfg` arguments and `--check-cfg` is available
[down below](#equivalence-table-with---cfg).

## Well known names and values

`rustc` maintains a list of well-known names and their corresponding values in order to avoid
the need to specify them manually.

Well known names and values are implicitly added as long as at least one `--check-cfg` argument
is present.

As of `2025-01-02T`, the list of known names is as follows:

<!--- See CheckCfg::fill_well_known in compiler/rustc_session/src/config.rs -->

 - `clippy`
 - `debug_assertions`
 - `doc`
 - `doctest`
 - `fmt_debug`
 - `miri`
 - `overflow_checks`
 - `panic`
 - `proc_macro`
 - `relocation_model`
 - `rustfmt`
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
 - `ub_checks`
 - `unix`
 - `windows`

> Starting with 1.85.0, the `test` cfg is considered to be a "userspace" config
> despite being also set by `rustc` and should be managed by the build system itself.

Like with `values(any())`, well known names checking can be disabled by passing `cfg(any())`
as argument to `--check-cfg`.

## Equivalence table with `--cfg`

This table describe the equivalence between a `--cfg` argument to a `--check-cfg` argument.

| `--cfg`                       | `--check-cfg`                                              |
|-------------------------------|------------------------------------------------------------|
| *nothing*                     | *nothing* or `--check-cfg=cfg()` (to enable the checking)  |
| `--cfg foo`                   | `--check-cfg=cfg(foo)` or `--check-cfg=cfg(foo, values(none()))` |
| `--cfg foo=""`                | `--check-cfg=cfg(foo, values(""))`                         |
| `--cfg foo="bar"`             | `--check-cfg=cfg(foo, values("bar"))`                      |
| `--cfg foo="1" --cfg foo="2"` | `--check-cfg=cfg(foo, values("1", "2"))`                   |
| `--cfg foo="1" --cfg bar="2"` | `--check-cfg=cfg(foo, values("1")) --check-cfg=cfg(bar, values("2"))` |
| `--cfg foo --cfg foo="bar"`   | `--check-cfg=cfg(foo, values(none(), "bar"))`              |

## Examples

### Example: Cargo-like `feature` example

Consider this command line:

```bash
rustc --check-cfg 'cfg(feature, values("lion", "zebra"))' \
      --cfg 'feature="lion"' example.rs
```

> This command line indicates that this crate has two features: `lion` and `zebra`. The `lion`
feature is enabled, while the `zebra` feature is disabled.

```rust
#[cfg(feature = "lion")]     // This condition is expected, as "lion" is an
                             // expected value of `feature`
fn tame_lion(lion: Lion) {}

#[cfg(feature = "zebra")]    // This condition is expected, as "zebra" is an expected
                             // value of `feature` but the condition will evaluate
                             // to false since only --cfg feature="lion" was passed
fn ride_zebra(z: Zebra) {}

#[cfg(feature = "platypus")] // This condition is UNEXPECTED, as "platypus" is NOT
                             // an expected value of `feature` and will cause a
                             // the compiler to emit the `unexpected_cfgs` lint
fn poke_platypus() {}

#[cfg(feechure = "lion")]    // This condition is UNEXPECTED, as 'feechure' is NOT
                             // a expected condition name, no `cfg(feechure, ...)`
                             // was passed in `--check-cfg`
fn tame_lion() {}

#[cfg(windows = "unix")]     // This condition is UNEXPECTED, as the well known
                             // 'windows' cfg doesn't expect any values
fn tame_windows() {}
```

### Example: Multiple names and values

```bash
rustc --check-cfg 'cfg(is_embedded, has_feathers)' \
      --check-cfg 'cfg(feature, values("zapping", "lasers"))' \
      --cfg has_feathers --cfg 'feature="zapping"'
```

```rust
#[cfg(is_embedded)]         // This condition is expected, as 'is_embedded' was
                            // provided in --check-cfg and doesn't take any value
fn do_embedded() {}

#[cfg(has_feathers)]        // This condition is expected, as 'has_feathers' was
                            // provided in --check-cfg and doesn't take any value
fn do_features() {}

#[cfg(has_mumble_frotz)]    // This condition is UNEXPECTED, as 'has_mumble_frotz'
                            // was NEVER provided in any --check-cfg arguments
fn do_mumble_frotz() {}

#[cfg(feature = "lasers")]  // This condition is expected, as "lasers" is an
                            // expected value of `feature`
fn shoot_lasers() {}

#[cfg(feature = "monkeys")] // This condition is UNEXPECTED, as "monkeys" is NOT
                            // an expected value of `feature`
fn write_shakespeare() {}
```

### Example: Condition names without values

```bash
rustc --check-cfg 'cfg(is_embedded, has_feathers, values(any()))' \
      --cfg has_feathers
```

```rust
#[cfg(is_embedded)]      // This condition is expected, as 'is_embedded' was
                         // provided in --check-cfg as condition name
fn do_embedded() {}

#[cfg(has_feathers)]     // This condition is expected, as "has_feathers" was
                         // provided in --check-cfg as condition name
fn do_features() {}

#[cfg(has_feathers = "zapping")] // This condition is expected, as "has_feathers"
                                 // was provided and because *any* values is
                                 // expected for 'has_feathers' no
                                 // warning is emitted for the value "zapping"
fn do_zapping() {}

#[cfg(has_mumble_frotz)] // This condition is UNEXPECTED, as 'has_mumble_frotz'
                         // was not provided in any --check-cfg arguments
fn do_mumble_frotz() {}
```
