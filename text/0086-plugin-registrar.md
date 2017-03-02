- Start Date: 2014-05-22
- RFC PR: [rust-lang/rfcs#86](https://github.com/rust-lang/rfcs/pull/86)
- Rust Issue: [rust-lang/rust#14637](https://github.com/rust-lang/rust/issues/14637)

# Summary

Generalize the `#[macro_registrar]` feature so it can register other kinds of compiler plugins.

# Motivation

I want to implement [loadable lints](https://github.com/mozilla/rust/issues/14067) and use them for project-specific static analysis passes in Servo.  Landing this first will allow more evolution of the plugin system without breaking source compatibility for existing users.

# Detailed design

To register a procedural macro in current Rust:

~~~ .rs
use syntax::ast::Name;
use syntax::parse::token;
use syntax::ext::base::{SyntaxExtension, BasicMacroExpander, NormalTT};

#[macro_registrar]
pub fn macro_registrar(register: |Name, SyntaxExtension|) {
    register(token::intern("named_entities"),
        NormalTT(box BasicMacroExpander {
            expander: named_entities::expand,
            span: None
        },
        None));
}
~~~

I propose an interface like

~~~ .rs
use syntax::parse::token;
use syntax::ext::base::{BasicMacroExpander, NormalTT};

use rustc::plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_macro(token::intern("named_entities"),
        NormalTT(box BasicMacroExpander {
            expander: named_entities::expand,
            span: None
        },
        None));
}
~~~

Then the struct `Registry` could provide additional methods such as `register_lint` as those features are implemented.

It could also provide convenience methods:

~~~ .rs
use rustc::plugin::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_simple_macro("named_entities", named_entities::expand);
}
~~~

`phase(syntax)` becomes `phase(plugin)`, with the former as a deprecated synonym that warns.  This is to avoid silent breakage of the very common `#[phase(syntax)] extern crate log`.

We only need one phase of loading plugin crates, even though the plugins we load may be used at different points (or not at all).

# Drawbacks

Breaking change for existing procedural macros.

More moving parts.

`Registry` is provided by `librustc`, because it will have methods for registering lints and other `librustc` things.  This means that syntax extensions must link `librustc`, when before they only needed `libsyntax` (but could link `librustc` anyway if desired).  This was discussed [on the RFC PR](https://github.com/rust-lang/rfcs/pull/86) and [the Rust PR](https://github.com/mozilla/rust/pull/14554) and [on IRC](https://botbot.me/mozilla/rust-internals/2014-05-22/?msg=15075433&page=5).

`#![feature(macro_registrar)]` becomes unknown, contradicting a comment in `feature_gate.rs`:

> This list can never shrink, it may only be expanded (in order to prevent old programs from failing to compile)

Since when do we ensure that old programs will compile? ;)  The `#[macro_registrar]` attribute wouldn't work anyway.

# Alternatives

We could add `#[lint_registrar]` etc. alongside `#[macro_registrar]`.  This seems like it will produce more duplicated effort all around.  It doesn't provide convenience methods, and it won't support API evolution as well.

We could support the old `#[macro_registrar]` by injecting an adapter shim.  This is significant extra work to support a feature with no stability guarantee.

# Unresolved questions

Naming bikeshed.

What set of convenience methods should we provide?
