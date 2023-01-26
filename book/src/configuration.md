# Configuring Clippy

> **Note:** The configuration file is unstable and may be deprecated in the future.

Some lints can be configured in a TOML file named `clippy.toml` or `.clippy.toml`. It contains a
basic `variable = value` mapping eg.

```toml
avoid-breaking-exported-api = false
disallowed-names = ["toto", "tata", "titi"]
```

The [table of configurations](./lint_configuration.md)
contains all config values, their default, and a list of lints they affect.
Each [configurable lint](https://rust-lang.github.io/rust-clippy/master/index.html#Configuration)
, also contains information about these values.

For configurations that are a list type with default values such as
[disallowed-names](https://rust-lang.github.io/rust-clippy/master/index.html#disallowed_names),
you can use the unique value `".."` to extend the default values instead of replacing them.

```toml
# default of disallowed-names is ["foo", "baz", "quux"]
disallowed-names = ["bar", ".."] # -> ["bar", "foo", "baz", "quux"]
```

To deactivate the "for further information visit *lint-link*" message you can define the `CLIPPY_DISABLE_DOCS_LINKS`
environment variable.

### Allowing/denying lints

You can add options to your code to `allow`/`warn`/`deny` Clippy lints:

* the whole set of `Warn` lints using the `clippy` lint group (`#![deny(clippy::all)]`)

* all lints using both the `clippy` and `clippy::pedantic` lint groups (`#![deny(clippy::all)]`,
  `#![deny(clippy::pedantic)]`). Note that `clippy::pedantic` contains some very aggressive lints prone to false
  positives.

* only some lints (`#![deny(clippy::single_match, clippy::box_vec)]`, etc.)

* `allow`/`warn`/`deny` can be limited to a single function or module using `#[allow(...)]`, etc.

Note: `allow` means to suppress the lint for your code. With `warn` the lint will only emit a warning, while with `deny`
the lint will emit an error, when triggering for your code. An error causes clippy to exit with an error code, so is
useful in scripts like CI/CD.

If you do not want to include your lint levels in your code, you can globally enable/disable lints by passing extra
flags to Clippy during the run:

To allow `lint_name`, run

```terminal
cargo clippy -- -A clippy::lint_name
```

And to warn on `lint_name`, run

```terminal
cargo clippy -- -W clippy::lint_name
```

This also works with lint groups. For example you can run Clippy with warnings for all lints enabled:

```terminal
cargo clippy -- -W clippy::pedantic
```

If you care only about a single lint, you can allow all others and then explicitly warn on the lint(s) you are
interested in:

```terminal
cargo clippy -- -A clippy::all -W clippy::useless_format -W clippy::...
```

### Specifying the minimum supported Rust version

Projects that intend to support old versions of Rust can disable lints pertaining to newer features by specifying the
minimum supported Rust version (MSRV) in the clippy configuration file.

```toml
msrv = "1.30.0"
```

The MSRV can also be specified as an attribute, like below.

```rust
#![feature(custom_inner_attributes)]
#![clippy::msrv = "1.30.0"]

fn main() {
    ...
}
```

You can also omit the patch version when specifying the MSRV, so `msrv = 1.30`
is equivalent to `msrv = 1.30.0`.

Note: `custom_inner_attributes` is an unstable feature so it has to be enabled explicitly.

Lints that recognize this configuration option can be
found [here](https://rust-lang.github.io/rust-clippy/master/index.html#msrv)

### Disabling evaluation of certain code

> **Note:** This should only be used in cases where other solutions, like `#[allow(clippy::all)]`, are not sufficient.

Very rarely, you may wish to prevent Clippy from evaluating certain sections of code entirely. You can do this with
[conditional compilation](https://doc.rust-lang.org/reference/conditional-compilation.html) by checking that the
`cargo-clippy` feature is not set. You may need to provide a stub so that the code compiles:

```rust
#[cfg(not(feature = "cargo-clippy"))]
include!(concat!(env!("OUT_DIR"), "/my_big_function-generated.rs"));

#[cfg(feature = "cargo-clippy")]
fn my_big_function(_input: &str) -> Option<MyStruct> {
    None
}
```

This feature is not actually part of your crate, so specifying `--all-features` to other tools, e.g. `cargo test
--all-features`, will not disable it.
