//! Centralized logic for parsing and validating all attributes used after HIR.
//!
//! History: Check out [#131229](https://github.com/rust-lang/rust/issues/131229).
//! There used to be only one definition of attributes in the compiler: `ast::Attribute`.
//! These were then parsed or validated or both in places distributed all over the compiler.
//!
//! Attributes are markers on items. Most are actually attribute-like proc-macros, and are expanded
//! but some remain as the built-in attributes to guide compilation.
//!
//! In this crate, syntactical attributes (sequences of tokens that look like
//! `#[something(something else)]`) are parsed into more semantic attributes, markers on items.
//! Multiple syntactic attributes might influence a single semantic attribute. For example,
//! `#[stable(...)]` and `#[unstable()]` cannot occur together, and both semantically define
//! a "stability". Stability defines an [`AttributeExtractor`](attributes::AttributeExtractor)
//! that recognizes both `#[stable()]` and `#[unstable()]` syntactic attributes, and at the end
//! produce a single [`ParsedAttributeKind::Stability`].
//!
//! FIXME(jdonszelmann): update devguide for best practices on attributes
//! FIXME(jdonszelmann): rename to `rustc_attr` in the future, integrating it into this crate.
//!
//! To define a new builtin, first add it

// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(rust_logo)]
#![feature(let_chains)]
#![feature(rustdoc_internals)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

#[macro_use]
mod attributes;
mod context;
mod parser;
mod session_diagnostics;

pub use attributes::cfg::*;
pub use attributes::util::{find_crate_name, is_builtin_attr, parse_version};
pub use context::AttributeParseContext;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

#[macro_export]
macro_rules! find_attr {
    ($attributes_list: expr, $pattern: pat $(if $guard: expr)?) => {{
        $crate::find_attr!($attributes_list, $pattern $(if $guard)? => ()).is_some()
    }};

    ($attributes_list: expr, $pattern: pat $(if $guard: expr)? => $e: expr) => {{
        fn check_attribute_iterator<'a>(_: &'_ impl IntoIterator<Item = &'a rustc_hir::Attribute>) {}
        check_attribute_iterator(&$attributes_list);

        let find_attribute = |iter| {
            for i in $attributes_list {
                match i {
                    rustc_hir::Attribute::Parsed($pattern) $(if $guard: expr)? => {
                        return Some($e);
                    }
                    _ => {}
                }
            }

            None
        };
        find_attribute($attributes_list)
    }};
}
