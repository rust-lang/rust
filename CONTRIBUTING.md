# Contributing to rust-clippy

Hello fellow Rustacean! Great to see your interest in compiler internals and lints!

Clippy welcomes contributions from everyone. There are many ways to contribute to Clippy and the following document explains how
you can contribute and how to get started.
If you have any questions about contributing or need help with anything, feel free to ask questions on issues or
visit the `#clippy` IRC channel on `irc.mozilla.org`.

All contributors are expected to follow the [Rust Code of Conduct](http://www.rust-lang.org/conduct.html).

* [Getting started](#getting-started)
  * [Finding something to fix/improve](#finding-something-to-fiximprove)
* [Writing code](#writing-code)
  * [Author lint](#author-lint)
  * [Documentation](#documentation)
  * [Running test suite](#running-test-suite)
  * [Testing manually](#testing-manually)
  * [How Clippy works](#how-clippy-works)
* [Contributions](#contributions)

## Getting started

High level approach:

1. Find something to fix/improve
2. Change code (likely some file in `clippy_lints/src/`)
3. Run `cargo test` in the root directory and wiggle code until it passes
4. Open a PR (also can be done between 2. and 3. if you run into problems)

### Finding something to fix/improve

All issues on Clippy are mentored, if you want help with a bug just ask @Manishearth, @llogiq, @mcarton or @oli-obk.

Some issues are easier than others. The [`good first issue`](https://github.com/rust-lang-nursery/rust-clippy/labels/good%20first%20issue)
label can be used to find the easy issues. If you want to work on an issue, please leave a comment
so that we can assign it to you!

Issues marked [`T-AST`](https://github.com/rust-lang-nursery/rust-clippy/labels/T-AST) involve simple
matching of the syntax tree structure, and are generally easier than
[`T-middle`](https://github.com/rust-lang-nursery/rust-clippy/labels/T-middle) issues, which involve types
and resolved paths.

[`T-AST`](https://github.com/rust-lang-nursery/rust-clippy/labels/T-AST) issues will generally need you to match against a predefined syntax structure. To figure out
how this syntax structure is encoded in the AST, it is recommended to run `rustc -Z ast-json` on an
example of the structure and compare with the
[nodes in the AST docs](https://doc.rust-lang.org/nightly/nightly-rustc/syntax/ast). Usually
the lint will end up to be a nested series of matches and ifs,
[like so](https://github.com/rust-lang-nursery/rust-clippy/blob/de5ccdfab68a5e37689f3c950ed1532ba9d652a0/src/misc.rs#L34).

[`E-medium`](https://github.com/rust-lang-nursery/rust-clippy/labels/E-medium) issues are generally
pretty easy too, though it's recommended you work on an E-easy issue first. They are mostly classified
as `E-medium`, since they might be somewhat involved code wise, but not difficult per-se.

[`T-middle`](https://github.com/rust-lang-nursery/rust-clippy/labels/T-middle) issues can
be more involved and require verifying types. The
[`ty`](https://doc.rust-lang.org/nightly/nightly-rustc/rustc/ty) module contains a
lot of methods that are useful, though one of the most useful would be `expr_ty` (gives the type of
an AST expression). `match_def_path()` in Clippy's `utils` module can also be useful.

## Writing code

Compiling clippy from scratch can take almost a minute or more depending on your machine.
However, since Rust 1.24.0 incremental compilation is enabled by default and compile times for small changes should be quick.

[Llogiq's blog post on lints](https://llogiq.github.io/2015/06/04/workflows.html) is a nice primer
to lint-writing, though it does get into advanced stuff. Most lints consist of an implementation of
`LintPass` with one or more of its default methods overridden. See the existing lints for examples
of this.


### Author lint

There is also the internal `author` lint to generate clippy code that detects the offending pattern. It does not work for all of the Rust syntax, but can give a good starting point.

First, create a new UI test file in the `tests/ui/` directory with the pattern you want to match:

```rust
// ./tests/ui/my_lint.rs

// The custom_attribute needs to be enabled for the author lint to work
#![feature(plugin, custom_attribute)]

fn main() {
    #[clippy(author)]
    let arr: [i32; 1] = [7]; // Replace line with the code you want to match
}
```

Now you run `TESTNAME=ui/my_lint cargo test --test compile-test` to produce
a `.stdout` file with the generated code:

```rust
// ./tests/ui/my_lint.stdout

if_chain! {
    if let Expr_::ExprArray(ref elements) = stmt.node;
    if elements.len() == 1;
    if let Expr_::ExprLit(ref lit) = elements[0].node;
    if let LitKind::Int(7, _) = lit.node;
    then {
        // report your lint here
    }
}
```

If the command was executed successfully, you can copy the code over to where you are implementing your lint.

### Documentation

Please document your lint with a doc comment akin to the following:

```rust
/// **What it does:** Checks for ... (describe what the lint matches).
///
/// **Why is this bad?** Supply the reason for linting the code.
///
/// **Known problems:** None. (Or describe where it could go wrong.)
///
/// **Example:**
///
/// ```rust
/// // Bad
/// Insert a short example of code that triggers the lint
///
/// // Good
/// Insert a short example of improved code that doesn't trigger the lint
/// ```
```

Once your lint is merged it will show up in the [lint list](https://rust-lang-nursery.github.io/rust-clippy/master/index.html)

### Running test suite

Use `cargo test` to run the whole testsuite.

If you don't want to wait for all tests to finish, you can also execute a single test file by using `TESTNAME` to specify the test to run:

```bash
TESTNAME=ui/empty_line_after_outer_attr cargo test --test compile-test
```

Clippy uses UI tests. UI tests check that the output of the compiler is exactly as expected.
Of course there's little sense in writing the output yourself or copying it around.
Therefore you should use `tests/ui/update-all-references.sh` (after running
`cargo test`) and check whether the output looks as you expect with `git diff`. Commit all
`*.stderr` files, too.

### Testing manually

Manually testing against an example file is useful if you have added some
`println!`s and test suite output becomes unreadable.  To try clippy with your
local modifications, run `cargo run --bin clippy-driver -- -L ./target/debug input.rs` from the
working copy root. Your test file, here `input.rs`, needs to have clippy
enabled as a plugin:

```rust
#![feature(plugin)]
#![plugin(clippy)]
```

### How Clippy works

Clippy is a [rustc compiler plugin][compiler_plugin]. The main entry point is at [`src/lib.rs`][main_entry]. In there, the lint registration is delegated to the [`clippy_lints`][lint_crate] crate.

[`clippy_lints/src/lib.rs`][lint_crate_entry] imports all the different lint modules and registers them with the rustc plugin registry. For example, the [`else_if_without_else`][else_if_without_else] lint is registered like this:

```rust
// ./clippy_lints/src/lib.rs

// ...
pub mod else_if_without_else;
// ...

pub fn register_plugins(reg: &mut rustc_plugin::Registry) {
    // ...
    reg.register_early_lint_pass(box else_if_without_else::ElseIfWithoutElse);
    // ...

    reg.register_lint_group("clippy_restriction", vec![
        // ...
        else_if_without_else::ELSE_IF_WITHOUT_ELSE,
        // ...
    ]);
}
```

The [`rustc_plugin::PluginRegistry`][plugin_registry] provides two methods to register lints: [register_early_lint_pass][reg_early_lint_pass] and [register_late_lint_pass][reg_late_lint_pass].
Both take an object that implements an [`EarlyLintPass`][early_lint_pass] or [`LateLintPass`][late_lint_pass] respectively. This is done in every single lint. 
It's worth noting that the majority of `clippy_lints/src/lib.rs` is autogenerated by `util/update_lints.py` and you don't have to add anything by hand. When you are writing your own lint, you can use that script to save you some time.

```rust
// ./clippy_lints/src/else_if_without_else.rs

use rustc::lint::*;

// ...

pub struct ElseIfWithoutElse;

// ...

impl EarlyLintPass for ElseIfWithoutElse {
    // ... the functions needed, to make the lint work
}
```

The difference between `EarlyLintPass` and `LateLintPass` is that the methods of the `EarlyLintPass` trait only provide AST information. The methods of the `LateLintPass` trait are executed after type checking and contain type information via the `LateContext` parameter.

That's why the `else_if_without_else` example uses the `register_early_lint_pass` function. Because the [actual lint logic][else_if_without_else] does not depend on any type information.

## Contributions

Contributions to Clippy should be made in the form of GitHub pull requests. Each pull request will
be reviewed by a core contributor (someone with permission to land patches) and either landed in the
main tree or given feedback for changes that would be required.

All code in this repository is under the [Mozilla Public License, 2.0](https://www.mozilla.org/MPL/2.0/)

<!-- adapted from https://github.com/servo/servo/blob/master/CONTRIBUTING.md -->

[main_entry]: https://github.com/rust-lang-nursery/rust-clippy/blob/c5b39a5917ffc0f1349b6e414fa3b874fdcf8429/src/lib.rs#L14
[lint_crate]: https://github.com/rust-lang-nursery/rust-clippy/tree/c5b39a5917ffc0f1349b6e414fa3b874fdcf8429/clippy_lints/src
[lint_crate_entry]: https://github.com/rust-lang-nursery/rust-clippy/blob/c5b39a5917ffc0f1349b6e414fa3b874fdcf8429/clippy_lints/src/lib.rs
[else_if_without_else]: https://github.com/rust-lang-nursery/rust-clippy/blob/c5b39a5917ffc0f1349b6e414fa3b874fdcf8429/clippy_lints/src/else_if_without_else.rs
[compiler_plugin]: https://doc.rust-lang.org/unstable-book/language-features/plugin.html#lint-plugins
[plugin_registry]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_plugin/registry/struct.Registry.html
[reg_early_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_plugin/registry/struct.Registry.html#method.register_early_lint_pass
[reg_late_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_plugin/registry/struct.Registry.html#method.register_late_lint_pass
[early_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/lint/trait.EarlyLintPass.html
[late_lint_pass]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc/lint/trait.LateLintPass.html
