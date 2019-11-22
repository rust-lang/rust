# `plugin`

The tracking issue for this feature is: [#29597]

[#29597]: https://github.com/rust-lang/rust/issues/29597


This feature is part of "compiler plugins." It will often be used with the
[`plugin_registrar`] and `rustc_private` features.

[`plugin_registrar`]: plugin-registrar.md

------------------------

`rustc` can load compiler plugins, which are user-provided libraries that
extend the compiler's behavior with new lint checks, etc.

A plugin is a dynamic library crate with a designated *registrar* function that
registers extensions with `rustc`. Other crates can load these extensions using
the crate attribute `#![plugin(...)]`.  See the
`rustc_driver::plugin` documentation for more about the
mechanics of defining and loading a plugin.

If present, arguments passed as `#![plugin(foo(... args ...))]` are not
interpreted by rustc itself.  They are provided to the plugin through the
`Registry`'s `args` method.

In the vast majority of cases, a plugin should *only* be used through
`#![plugin]` and not through an `extern crate` item.  Linking a plugin would
pull in all of libsyntax and librustc as dependencies of your crate.  This is
generally unwanted unless you are building another plugin.  The
`plugin_as_library` lint checks these guidelines.

The usual practice is to put compiler plugins in their own crate, separate from
any `macro_rules!` macros or ordinary Rust code meant to be used by consumers
of a library.

# Lint plugins

Plugins can extend [Rust's lint
infrastructure](../../reference/attributes/diagnostics.md#lint-check-attributes) with
additional checks for code style, safety, etc. Now let's write a plugin
[`lint_plugin_test.rs`](https://github.com/rust-lang/rust/blob/master/src/test/ui-fulldeps/auxiliary/lint_plugin_test.rs)
that warns about any item named `lintme`.

```rust,ignore
#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate syntax;

// Load rustc as a plugin to get macros
#[macro_use]
extern crate rustc;
extern crate rustc_driver;

use rustc::lint::{EarlyContext, LintContext, LintPass, EarlyLintPass,
                  EarlyLintPassObject, LintArray};
use rustc_driver::plugin::Registry;
use syntax::ast;

declare_lint!(TEST_LINT, Warn, "Warn about items named 'lintme'");

struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TEST_LINT)
    }
}

impl EarlyLintPass for Pass {
    fn check_item(&mut self, cx: &EarlyContext, it: &ast::Item) {
        if it.ident.as_str() == "lintme" {
            cx.span_lint(TEST_LINT, it.span, "item is named 'lintme'");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_early_lint_pass(box Pass as EarlyLintPassObject);
}
```

Then code like

```rust,ignore
#![plugin(lint_plugin_test)]

fn lintme() { }
```

will produce a compiler warning:

```txt
foo.rs:4:1: 4:16 warning: item is named 'lintme', #[warn(test_lint)] on by default
foo.rs:4 fn lintme() { }
         ^~~~~~~~~~~~~~~
```

The components of a lint plugin are:

* one or more `declare_lint!` invocations, which define static `Lint` structs;

* a struct holding any state needed by the lint pass (here, none);

* a `LintPass`
  implementation defining how to check each syntax element. A single
  `LintPass` may call `span_lint` for several different `Lint`s, but should
  register them all through the `get_lints` method.

Lint passes are syntax traversals, but they run at a late stage of compilation
where type information is available. `rustc`'s [built-in
lints](https://github.com/rust-lang/rust/blob/master/src/librustc/lint/builtin.rs)
mostly use the same infrastructure as lint plugins, and provide examples of how
to access type information.

Lints defined by plugins are controlled by the usual [attributes and compiler
flags](../../reference/attributes/diagnostics.md#lint-check-attributes), e.g.
`#[allow(test_lint)]` or `-A test-lint`. These identifiers are derived from the
first argument to `declare_lint!`, with appropriate case and punctuation
conversion.

You can run `rustc -W help foo.rs` to see a list of lints known to `rustc`,
including those provided by plugins loaded by `foo.rs`.
