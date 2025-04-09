//! Regression test for #128622.
//!
//! PR #128581 introduced an assertion that all builtin attributes are actually checked via
//! `CheckAttrVisitor` and aren't accidentally usable on completely unrelated HIR nodes.
//! Unfortunately, the check had correctness problems.
//!
//! The match on attribute path segments looked like
//!
//! ```rs,ignore
//! [sym::should_panic] => /* check is implemented */
//! match BUILTIN_ATTRIBUTE_MAP.get(name) {
//!     // checked below
//!     Some(BuiltinAttribute { type_: AttributeType::CrateLevel, .. }) => {}
//!     Some(_) => {
//!         if !name.as_str().starts_with("rustc_") {
//!             span_bug!(
//!                 attr.span,
//!                 "builtin attribute {name:?} not handled by `CheckAttrVisitor`"
//!             )
//!         }
//!     }
//!     None => (),
//! }
//! ```
//!
//! However, it failed to account for edge cases such as an attribute whose:
//!
//! 1. path segments *starts* with a builtin attribute such as `should_panic`
//! 2. which does not start with `rustc_`, and
//! 3. is also an `AttributeType::Normal` attribute upon registration with the builtin attribute map
//!
//! These conditions when all satisfied cause the span bug to be issued for e.g.
//! `#[should_panic::skip]` because the `[sym::should_panic]` arm is not matched (since it's
//! `[sym::should_panic, sym::skip]`).
//!
//! This test checks that the span bug is not fired for such cases.
//!
//! issue: rust-lang/rust#128622

// Notably, `should_panic` is a `AttributeType::Normal` attribute that is checked separately.

#![deny(unused_attributes)]

struct Foo {
    #[should_panic::skip]
    //~^ ERROR failed to resolve
    //~| ERROR `#[should_panic::skip]` only has an effect on functions
    pub field: u8,

    #[should_panic::a::b::c]
    //~^ ERROR failed to resolve
    //~| ERROR `#[should_panic::a::b::c]` only has an effect on functions
    pub field2: u8,
}

fn foo() {}

fn main() {
    #[deny::skip]
    //~^ ERROR failed to resolve
    foo();
}
