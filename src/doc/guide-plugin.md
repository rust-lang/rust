% The Rust Compiler Plugins Guide

<div class="unstable-feature">

<p>
<b>Warning:</b> Plugins are an advanced, unstable feature! For many details,
the only available documentation is the <a
href="syntax/index.html"><code>libsyntax</code></a> and <a
href="rustc/index.html"><code>librustc</code></a> API docs, or even the source
code itself. These internal compiler APIs are also subject to change at any
time.
</p>

<p>
For defining new syntax it is often much easier to use Rust's <a
href="guide-macros.html">built-in macro system</a>.
</p>

<p style="margin-bottom: 0">
The code in this document uses language features not covered in the Rust
Guide.  See the <a href="reference.html">Reference Manual</a> for more
information.
</p>

</div>

# Introduction

`rustc` can load compiler plugins, which are user-provided libraries that
extend the compiler's behavior with new syntax extensions, lint checks, etc.

A plugin is a dynamic library crate with a designated "registrar" function that
registers extensions with `rustc`. Other crates can use these extensions by
loading the plugin crate with `#[phase(plugin)] extern crate`. See the
[`rustc::plugin`](rustc/plugin/index.html) documentation for more about the
mechanics of defining and loading a plugin.

# Syntax extensions

Plugins can extend Rust's syntax in various ways. One kind of syntax extension
is the procedural macro. These are invoked the same way as [ordinary
macros](guide-macros.html), but the expansion is performed by arbitrary Rust
code that manipulates [syntax trees](syntax/ast/index.html) at
compile time.

Let's write a plugin
[`roman_numerals.rs`](https://github.com/rust-lang/rust/tree/master/src/test/auxiliary/roman_numerals.rs)
that implements Roman numeral integer literals.

```ignore
#![crate_type="dylib"]
#![feature(plugin_registrar)]

extern crate syntax;
extern crate rustc;

use syntax::codemap::Span;
use syntax::parse::token::{IDENT, get_ident};
use syntax::ast::{TokenTree, TTTok};
use syntax::ext::base::{ExtCtxt, MacResult, DummyResult, MacExpr};
use syntax::ext::build::AstBuilder;  // trait for expr_uint
use rustc::plugin::Registry;

fn expand_rn(cx: &mut ExtCtxt, sp: Span, args: &[TokenTree])
        -> Box<MacResult + 'static> {

    static NUMERALS: &'static [(&'static str, uint)] = &[
        ("M", 1000), ("CM", 900), ("D", 500), ("CD", 400),
        ("C",  100), ("XC",  90), ("L",  50), ("XL",  40),
        ("X",   10), ("IX",   9), ("V",   5), ("IV",   4),
        ("I",    1)];

    let text = match args {
        [TTTok(_, IDENT(s, _))] => get_ident(s).to_string(),
        _ => {
            cx.span_err(sp, "argument should be a single identifier");
            return DummyResult::any(sp);
        }
    };

    let mut text = text.as_slice();
    let mut total = 0u;
    while !text.is_empty() {
        match NUMERALS.iter().find(|&&(rn, _)| text.starts_with(rn)) {
            Some(&(rn, val)) => {
                total += val;
                text = text.slice_from(rn.len());
            }
            None => {
                cx.span_err(sp, "invalid Roman numeral");
                return DummyResult::any(sp);
            }
        }
    }

    MacExpr::new(cx.expr_uint(sp, total))
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro("rn", expand_rn);
}
```

Then we can use `rn!()` like any other macro:

```ignore
#![feature(phase)]

#[phase(plugin)]
extern crate roman_numerals;

fn main() {
    assert_eq!(rn!(MMXV), 2015);
}
```

The advantages over a simple `fn(&str) -> uint` are:

* The (arbitrarily complex) conversion is done at compile time.
* Input validation is also performed at compile time.
* It can be extended to allow use in patterns, which effectively gives
  a way to define new literal syntax for any data type.

In addition to procedural macros, you can define new
[`deriving`](reference.html#deriving)-like attributes and other kinds of
extensions.  See
[`Registry::register_syntax_extension`](rustc/plugin/registry/struct.Registry.html#method.register_syntax_extension)
and the [`SyntaxExtension`
enum](http://doc.rust-lang.org/syntax/ext/base/enum.SyntaxExtension.html).  For
a more involved macro example, see
[`src/libregex_macros/lib.rs`](https://github.com/rust-lang/rust/blob/master/src/libregex_macros/lib.rs)
in the Rust distribution.


## Tips and tricks

To see the results of expanding syntax extensions, run
`rustc --pretty expanded`. The output represents a whole crate, so you
can also feed it back in to `rustc`, which will sometimes produce better
error messages than the original compilation. Note that the
`--pretty expanded` output may have a different meaning if multiple
variables of the same name (but different syntax contexts) are in play
in the same scope. In this case `--pretty expanded,hygiene` will tell
you about the syntax contexts.

You can use [`syntax::parse`](syntax/parse/index.html) to turn token trees into
higher-level syntax elements like expressions:

```ignore
fn expand_foo(cx: &mut ExtCtxt, sp: Span, args: &[TokenTree])
        -> Box<MacResult+'static> {

    let mut parser =
        parse::new_parser_from_tts(cx.parse_sess(), cx.cfg(), args.to_slice())

    let expr: P<Expr> = parser.parse_expr();
```

Looking through [`libsyntax` parser
code](https://github.com/rust-lang/rust/blob/master/src/libsyntax/parse/parser.rs)
will give you a feel for how the parsing infrastructure works.

Keep the [`Span`s](syntax/codemap/struct.Span.html) of
everything you parse, for better error reporting. You can wrap
[`Spanned`](syntax/codemap/struct.Spanned.html) around
your custom data structures.

Calling
[`ExtCtxt::span_fatal`](syntax/ext/base/struct.ExtCtxt.html#method.span_fatal)
will immediately abort compilation. It's better to instead call
[`ExtCtxt::span_err`](syntax/ext/base/struct.ExtCtxt.html#method.span_err)
and return
[`DummyResult`](syntax/ext/base/struct.DummyResult.html),
so that the compiler can continue and find further errors.

The example above produced an integer literal using
[`AstBuilder::expr_uint`](syntax/ext/build/trait.AstBuilder.html#tymethod.expr_uint).
As an alternative to the `AstBuilder` trait, `libsyntax` provides a set of
[quasiquote macros](syntax/ext/quote/index.html).  They are undocumented and
very rough around the edges.  However, the implementation may be a good
starting point for an improved quasiquote as an ordinary plugin library.


# Lint plugins

Plugins can extend [Rust's lint
infrastructure](reference.html#lint-check-attributes) with additional checks for
code style, safety, etc. You can see
[`src/test/auxiliary/lint_plugin_test.rs`](https://github.com/rust-lang/rust/blob/master/src/test/auxiliary/lint_plugin_test.rs)
for a full example, the core of which is reproduced here:

```ignore
declare_lint!(TEST_LINT, Warn,
              "Warn about items named 'lintme'")

struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(TEST_LINT)
    }

    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        let name = token::get_ident(it.ident);
        if name.get() == "lintme" {
            cx.span_lint(TEST_LINT, it.span, "item is named 'lintme'");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_lint_pass(box Pass as LintPassObject);
}
```

Then code like

```ignore
#[phase(plugin)]
extern crate lint_plugin_test;

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
  [`Lint`](rustc/lint/struct.Lint.html) structs;

* a struct holding any state needed by the lint pass (here, none);

* a [`LintPass`](rustc/lint/trait.LintPass.html)
  implementation defining how to check each syntax element. A single
  `LintPass` may call `span_lint` for several different `Lint`s, but should
  register them all through the `get_lints` method.

Lint passes are syntax traversals, but they run at a late stage of compilation
where type information is available. `rustc`'s [built-in
lints](https://github.com/rust-lang/rust/blob/master/src/librustc/lint/builtin.rs)
mostly use the same infrastructure as lint plugins, and provide examples of how
to access type information.

Lints defined by plugins are controlled by the usual [attributes and compiler
flags](reference.html#lint-check-attributes), e.g. `#[allow(test_lint)]` or
`-A test-lint`. These identifiers are derived from the first argument to
`declare_lint!`, with appropriate case and punctuation conversion.

You can run `rustc -W help foo.rs` to see a list of lints known to `rustc`,
including those provided by plugins loaded by `foo.rs`.
