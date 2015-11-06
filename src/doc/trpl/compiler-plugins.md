% Compiler Plugins

# Introduction

`rustc` can load compiler plugins, which are user-provided libraries that
extend the compiler's behavior with new syntax extensions, lint checks, etc.

A plugin is a dynamic library crate with a designated *registrar* function that
registers extensions with `rustc`. Other crates can load these extensions using
the crate attribute `#![plugin(...)]`.  See the
[`rustc::plugin`](../rustc/plugin/index.html) documentation for more about the
mechanics of defining and loading a plugin.

If present, arguments passed as `#![plugin(foo(... args ...))]` are not
interpreted by rustc itself.  They are provided to the plugin through the
`Registry`'s [`args` method](../rustc/plugin/registry/struct.Registry.html#method.args).

In the vast majority of cases, a plugin should *only* be used through
`#![plugin]` and not through an `extern crate` item.  Linking a plugin would
pull in all of libsyntax and librustc as dependencies of your crate.  This is
generally unwanted unless you are building another plugin.  The
`plugin_as_library` lint checks these guidelines.

The usual practice is to put compiler plugins in their own crate, separate from
any `macro_rules!` macros or ordinary Rust code meant to be used by consumers
of a library.

# Syntax extensions

Plugins can extend Rust's syntax in various ways. One kind of syntax extension
is the procedural macro. These are invoked the same way as [ordinary
macros](macros.html), but the expansion is performed by arbitrary Rust
code that manipulates [syntax trees](../syntax/ast/index.html) at
compile time.

Let's write a plugin
[`roman_numerals.rs`](https://github.com/rust-lang/rust/tree/master/src/test/auxiliary/roman_numerals.rs)
that implements Roman numeral integer literals.

```ignore
#![crate_type="dylib"]
#![feature(plugin_registrar, rustc_private)]

extern crate syntax;
extern crate rustc;

use syntax::codemap::Span;
use syntax::parse::token;
use syntax::ast::TokenTree;
use syntax::ext::base::{ExtCtxt, MacResult, DummyResult, MacEager};
use syntax::ext::build::AstBuilder;  // trait for expr_usize
use rustc::plugin::Registry;

fn expand_rn(cx: &mut ExtCtxt, sp: Span, args: &[TokenTree])
        -> Box<MacResult + 'static> {

    static NUMERALS: &'static [(&'static str, u32)] = &[
        ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
        ("C",  100), ("XC",  90), ("L",  50), ("XL",  40),
        ("X",   10), ("IX",   9), ("V",   5), ("IV",   4),
        ("I",    1)];

    let text = match args {
        [TokenTree::Token(_, token::Ident(s, _))] => s.to_string(),
        _ => {
            cx.span_err(sp, "argument should be a single identifier");
            return DummyResult::any(sp);
        }
    };

    let mut text = &*text;
    let mut total = 0;
    while !text.is_empty() {
        match NUMERALS.iter().find(|&&(rn, _)| text.starts_with(rn)) {
            Some(&(rn, val)) => {
                total += val;
                text = &text[rn.len()..];
            }
            None => {
                cx.span_err(sp, "invalid Roman numeral");
                return DummyResult::any(sp);
            }
        }
    }

    MacEager::expr(cx.expr_u32(sp, total))
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("rn", expand_rn);
}
```

Then we can use `rn!()` like any other macro:

```ignore
#![feature(plugin)]
#![plugin(roman_numerals)]

fn main() {
    assert_eq!(rn!(MMXV), 2015);
}
```

The advantages over a simple `fn(&str) -> u32` are:

* The (arbitrarily complex) conversion is done at compile time.
* Input validation is also performed at compile time.
* It can be extended to allow use in patterns, which effectively gives
  a way to define new literal syntax for any data type.

In addition to procedural macros, you can define new
[`derive`](../reference.html#derive)-like attributes and other kinds of
extensions.  See
[`Registry::register_syntax_extension`](../rustc/plugin/registry/struct.Registry.html#method.register_syntax_extension)
and the [`SyntaxExtension`
enum](https://doc.rust-lang.org/syntax/ext/base/enum.SyntaxExtension.html).  For
a more involved macro example, see
[`regex_macros`](https://github.com/rust-lang/regex/blob/master/regex_macros/src/lib.rs).


## Tips and tricks

Some of the [macro debugging tips](macros.html#debugging-macro-code) are applicable.

You can use [`syntax::parse`](../syntax/parse/index.html) to turn token trees into
higher-level syntax elements like expressions:

```ignore
fn expand_foo(cx: &mut ExtCtxt, sp: Span, args: &[TokenTree])
        -> Box<MacResult+'static> {

    let mut parser = cx.new_parser_from_tts(args);

    let expr: P<Expr> = parser.parse_expr();
```

Looking through [`libsyntax` parser
code](https://github.com/rust-lang/rust/blob/master/src/libsyntax/parse/parser.rs)
will give you a feel for how the parsing infrastructure works.

Keep the [`Span`s](../syntax/codemap/struct.Span.html) of
everything you parse, for better error reporting. You can wrap
[`Spanned`](../syntax/codemap/struct.Spanned.html) around
your custom data structures.

Calling
[`ExtCtxt::span_fatal`](../syntax/ext/base/struct.ExtCtxt.html#method.span_fatal)
will immediately abort compilation. It's better to instead call
[`ExtCtxt::span_err`](../syntax/ext/base/struct.ExtCtxt.html#method.span_err)
and return
[`DummyResult`](../syntax/ext/base/struct.DummyResult.html),
so that the compiler can continue and find further errors.

To print syntax fragments for debugging, you can use
[`span_note`](../syntax/ext/base/struct.ExtCtxt.html#method.span_note) together
with
[`syntax::print::pprust::*_to_string`](https://doc.rust-lang.org/syntax/print/pprust/index.html#functions).

The example above produced an integer literal using
[`AstBuilder::expr_usize`](../syntax/ext/build/trait.AstBuilder.html#tymethod.expr_usize).
As an alternative to the `AstBuilder` trait, `libsyntax` provides a set of
[quasiquote macros](../syntax/ext/quote/index.html).  They are undocumented and
very rough around the edges.  However, the implementation may be a good
starting point for an improved quasiquote as an ordinary plugin library.


# Lint plugins

Plugins can extend [Rust's lint
infrastructure](../reference.html#lint-check-attributes) with additional checks for
code style, safety, etc. Now let's write a plugin [`lint_plugin_test.rs`](https://github.com/rust-lang/rust/blob/master/src/test/auxiliary/lint_plugin_test.rs)
that warns about any item named `lintme`.

```ignore
#![feature(plugin_registrar)]
#![feature(box_syntax, rustc_private)]

extern crate syntax;

// Load rustc as a plugin to get macros
#[macro_use]
extern crate rustc;

use rustc::lint::{EarlyContext, LintContext, LintPass, EarlyLintPass,
                  EarlyLintPassObject, LintArray};
use rustc::plugin::Registry;
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
        if it.ident.name.as_str() == "lintme" {
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

```ignore
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

* one or more `declare_lint!` invocations, which define static
  [`Lint`](../rustc/lint/struct.Lint.html) structs;

* a struct holding any state needed by the lint pass (here, none);

* a [`LintPass`](../rustc/lint/trait.LintPass.html)
  implementation defining how to check each syntax element. A single
  `LintPass` may call `span_lint` for several different `Lint`s, but should
  register them all through the `get_lints` method.

Lint passes are syntax traversals, but they run at a late stage of compilation
where type information is available. `rustc`'s [built-in
lints](https://github.com/rust-lang/rust/blob/master/src/librustc/lint/builtin.rs)
mostly use the same infrastructure as lint plugins, and provide examples of how
to access type information.

Lints defined by plugins are controlled by the usual [attributes and compiler
flags](../reference.html#lint-check-attributes), e.g. `#[allow(test_lint)]` or
`-A test-lint`. These identifiers are derived from the first argument to
`declare_lint!`, with appropriate case and punctuation conversion.

You can run `rustc -W help foo.rs` to see a list of lints known to `rustc`,
including those provided by plugins loaded by `foo.rs`.
