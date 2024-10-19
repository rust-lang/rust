//! Centralized logic for parsing and validating all attributes used after HIR.
//!
//! History: Check out [#131229](https://github.com/rust-lang/rust/issues/131229).
//! There used to be only one definition of attributes in the compiler: `ast::Attribute`.
//! These were then parsed or validated or both in places distributed all over the compiler.
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

mod builtin;
mod session_diagnostics;

use std::marker::PhantomData;

pub use IntType::*;
pub use ReprAttr::*;
pub use StabilityLevel::*;
pub use builtin::*;
pub use context::AttributeParseContext;
pub(crate) use rustc_session::HashStableContext;
use {rustc_ast as ast, rustc_hir as hir};

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

mod context;
mod parser;

pub enum MaybeParsedAttribute<'a> {
    Parsed(hir::ParsedAttributeKind),
    MustRemainUnparsed(&'a ast::NormalAttr),
}

pub trait AttributeCollection<'a> {
    fn iter(self) -> impl Iterator<Item = &'a hir::Attribute> + 'a;
    // pub fn filter_by_name<A: AttributeExt>(attrs: &[A], name: Symbol) -> impl Iterator<Item = &A> {
    //     attrs.iter().filter(move |attr| attr.has_name(name))
    // }
    //
    // pub fn find_by_name<A: AttributeExt>(attrs: &[A], name: Symbol) -> Option<&A> {
    //     filter_by_name(attrs, name).next()
    // }
    //
    // pub fn first_attr_value_str_by_name(attrs: &[impl AttributeExt], name: Symbol) -> Option<Symbol> {
    //     find_by_name(attrs, name).and_then(|attr| attr.value_str())
    // }
    //
    // pub fn contains_name(&self, name: Symbol) -> bool {
    //     find_by_name(attrs, name).is_some()
    // }
}

impl<'a> AttributeCollection<'a> for &'a [hir::Attribute] {
    fn iter(self) -> impl Iterator<Item = &'a hir::Attribute> + 'a {
        self.iter()
    }
}
impl<'a> AttributeCollection<'a> for &'a Vec<hir::Attribute> {
    fn iter(self) -> impl Iterator<Item = &'a hir::Attribute> + 'a {
        <[_]>::iter(self)
    }
}

pub struct AttributeIterator<'a, I>(pub I, PhantomData<&'a ()>);
impl<'a, I> AttributeIterator<'a, I> {
    pub fn new(i: I) -> Self {
        Self(i, PhantomData)
    }
}

impl<'a, I: Iterator<Item = &'a hir::Attribute>> Iterator for AttributeIterator<'a, I> {
    type Item = &'a hir::Attribute;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, I: Iterator<Item = &'a hir::Attribute> + 'a> AttributeCollection<'a>
    for AttributeIterator<'a, I>
{
    fn iter(self) -> impl Iterator<Item = &'a hir::Attribute> + 'a {
        self
    }
}
