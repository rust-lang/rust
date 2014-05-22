- Start Date: 2014-05-22
- RFC PR #:
- Rust Issue #:

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
use syntax::ext::Registry;
use syntax::ext::base::{BasicMacroExpander, NormalTT};

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
use syntax::ext::Registry;

#[plugin_registrar]
pub fn plugin_registrar(reg: &mut Registry) {
    reg.register_simple_macro("named_entities", named_entities::expand);
}
~~~

# Drawbacks

Breaking change for existing procedural macros.

More moving parts.

# Alternatives

We could add `#[lint_registrar]` etc. alongside `#[macro_registrar]`.  This seems like it will produce more duplicated effort all around.  It doesn't provide convenience methods, and it won't support API evolution as well.

# Unresolved questions

Naming bikeshed.

Should `Registry` be provided by `libsyntax`, when it's used to register more than just syntax extensions?

What set of convenience methods should we provide?
