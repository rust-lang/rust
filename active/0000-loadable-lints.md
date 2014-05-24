- Start Date: 2014-05-23
- RFC PR #:
- Rust Issue #:

# Summary

Allow users to load custom lints into `rustc`, similar to loadable syntax extensions.

# Motivation

There are many possibilities for user-defined static checking:

* Enforcing correct usage of Servo's [JS-managed pointers](https://github.com/mozilla/servo/blob/master/src/components/script/dom/bindings/js.rs)
* kballard's use case: checking that `rust-lua` functions which call `longjmp` never have destructors on stack variables
* Enforcing a company or project style guide
* Detecting common misuses of a library, e.g. expensive or non-idiomatic constructs
* In cryptographic code, annotating which variables contain secrets and then forbidding their use in variable-time operations or memory addressing

Existing project-specific static checkers include:

* A [Clang plugin](https://tecnocode.co.uk/2013/12/09/clang-plugin-for-glib-and-gnome/) that detects misuse of GLib and GObject
* A [GCC plugin](https://gcc-python-plugin.readthedocs.org/en/latest/cpychecker.html) (written in Python!) that detects misuse of the CPython extension API
* [Sparse](https://sparse.wiki.kernel.org/index.php/Main_Page), which checks Linux kernel code for issues such as mixing up userspace and kernel pointers (often exploitable for privilege escalation)

We should make it easy to build such tools and integrate them with an existing Rust project.

# Detailed design

In `rustc::middle::lint`:

~~~ .rs
pub trait Lint {
    fn check_item(&mut self, cx: &Context, it: &ast::Item) { }
    fn check_expr(&mut self, cx: &Context, e: &ast::Expr) { }
    ...
}
~~~

To define a lint:

~~~ .rs
#![crate_id="lipogram_lint"]
#![crate_type="dylib"]
#![feature(plugin_registrar)]

extern crate syntax;
extern crate rustc;

use syntax::ast;
use syntax::parse::token;
use rustc::middle::lint;
use rustc::middle::lint::{Lint, Context};
use rustc::plugin::Registry;

struct Lipogram;

impl Lint for Lipogram {
    fn check_item(&mut self, cx: &Context, it: &ast::Item) {
        let name = token::get_ident(it.ident).get();
        if name.contains_char('e') || name.contains_char('E') {
            cx.span_lint(it.span, "item name contains the letter 'e'");
        }
    }
}

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    // Register the lint, with a default level
    reg.register_lint("letter_e", lint::warn, box Lipogram as Box<Lint>);
}
~~~

To use the lint when compiling another crate:

~~~ .rs
#[phase(syntax)]
extern crate lipogram_lint;

struct Foo;

#[allow(letter_e)]
struct Hello;
~~~

The `rustc` flags `-W`, `-A`, `-D`, and `-F` will also work with user-defined lints.

Ideally we would convert built-in lints to this infrastructure, and eliminate or shrink the big `enum Lint`.  Note that `cx.span_lint` lost its argument identifying the lint; instead the lint visitor and context will know which lint it's invoking.

# Drawbacks

More complexity.  More coupling of user code to `rustc` internals (with no official stability guarantee, of course).

See [RFC PR #86](https://github.com/rust-lang/rfcs/pull/86) for more discussion of `#[plugin_registrar]`.  The simple implementation will also require syntax extensions to link against `librustc`, increasing compile time.

# Alternatives

We could provide a more generic mechanism for user-defined AST visitors.  This could support other use cases like code transformation.  But it would be harder to use, and harder to integrate with the lint infrastructure.

# Unresolved questions

Should `phase(syntax)` be renamed, if it's used for more than syntax extensions?  Or should there be a separate `phase(lint)`?

This feature really wants the [improved `unknown_attribute`](https://github.com/rust-lang/rfcs/blob/master/active/0002-attribute-usage.md) RFC, for custom attributes.  But I'm fine with using `-A unknown_attribute` temporarily.

How do we accommodate the existing use of `session.add_lint` from outside `rustc::middle::lint`?

Do we provide guarantees about visit order for a lint, or the order of multiple lints defined in the same crate?  Some lints may require multiple passes.

Should `rustc -W help` show user-defined lints?  It can't unless a crate filename is also given.
