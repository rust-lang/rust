// Test that (for now at least) we reject inert attributes[^1] on the "value" of key-value
// attributes (*).
//
// (*) unless the key-value attribute is "early". Attributes on macro calls are early for
// example and so are `#![crate_name]`, `#![crate_type]` & `#![recursion_limit]`. The latter
// lead to fatal errors hence they're test in separate files (see `key-value-nested-attr-*`).
//
// [^1]: As well as `cfg` attributes as they would be too footgun-y.

//@ proc-macro: replace.rs
//@ proc-macro: derive_macro_with_helper.rs
//@ edition: 2018..

#![feature(stmt_expr_attributes, proc_macro_hygiene)]

// Nested inert attribute that's applied to the correct target.
#![doc = #[allow(dead_code)] "content"] //~ ERROR this attribute is not allowed in this context

// Nested inert attribute that's applied to the wrong target.
#![doc = #[repr(C)] "content"] //~ ERROR this attribute is not allowed in this context

// Make sure we only blame (highlight) `#[allow(..)]` and `#[repr(..)]`, not however
// `#[replace::identity]` which is perfectly legal as it's an active attribute that's
// be expanded away before we get to semantically validate the `#[doc = ..]`.
#![doc =
    #[allow(dead_code)] //~ ERROR these attributes are not allowed in this context
    #[replace::identity]
    #[repr(C)]
    "content"
]

// Ensure we specifically blame the `#[allow(..)]` and not the surrounding `#[replace::with(..)]`.
#![doc =
    #[replace::with(
        #[allow(dead_code)] //~ ERROR this attribute is not allowed in this context
        "replacement"
    )]
    "content"
]

#![doc =
    #[cfg_attr(true,
        repr(Rust) //~ ERROR this attribute is not allowed in this context
    )]
    "content"
]

#![doc = #[cfg(false)] "content"] //~ ERROR `cfg` is not supported in this position

// Tool attribute.
#[clippy::immaterial = #[allow(dead_code)] "incorporeal"]
//~^ ERROR this attribute is not allowed in this context
#[clippy::insubstantial = #[cfg(true)] "insignificant"]
//~^ ERROR `cfg` is not supported in this position
mod target {}

// Helper attribute.
#[derive(derive_macro_with_helper::Derive)]
#[arg = #[allow(dead_code)] 0] //~ ERROR this attribute is not allowed in this context
struct Owner;

// Ensure that we actually semantically validate the expansion: While we syntactically allow
// arbitrary expressions on the RHS, semantically we only allow unsuffixed literals
// (without any inert attributes).
#[doc = #[replace::with(2 * 4)] 8] //~ ERROR attribute value must be a literal
const _: () = ();

macro_rules! vanish {
    () => {};
}

// The key-value attribute doc is "early" because it's placed on a macro call. This reaches a
// different code path in the compiler. It means we reach the validation of `#[doc]` at a stage when
// the inner `#[replace::with(..)]` hasn't been expanded yet. What's more, it would never get
// expanded anyway since the whole attribute gets dropped early. Yes, it's even an active attribute.
// So let's flag out reject them (for now at least).
#[doc = #[replace::with("that")] "this"] //~ ERROR this attribute is not allowed in this context
vanish!();

fn main() {}
