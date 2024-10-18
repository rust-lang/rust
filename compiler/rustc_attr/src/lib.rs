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
use rustc_errors::DiagCtxtHandle;
pub(crate) use rustc_session::HashStableContext;
use rustc_span::ErrorGuaranteed;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

use rustc_ast as ast;
use rustc_hir::attribute::{Attribute, ParsedAttributeKind};

pub enum MaybeParsedAttribute<'a> {
    Parsed(ParsedAttributeKind),
    MustRemainUnparsed(&'a ast::NormalAttr),
}

pub fn parse_attribute_list<'a>(
    dcx: DiagCtxtHandle<'a>,
    attrs: &'a [ast::Attribute],
) -> impl Iterator<Item = (MaybeParsedAttribute<'a>, &'a ast::Attribute)> + use<'a> {
    attrs
        .iter()
        .map(move |attr| (parse_attribute(dcx, attr), attr))
        .filter_map(|(r, a)| Some((r.ok()?, a)))
}

/// Parses an attribute, if it can.
///
/// All attributes go through here to be parsed.
/// This function should return [`MaybeParsedAttribute::MustRemainUnparsed`] as little as possible,
/// because any time it does it implies that the attribute needs to be parsed somewhere else while
/// what we want is to centralize parsing.
///
/// Only [custom tool attributes](https://github.com/rust-lang/rust/issues/66079) can definitely not
/// be parsed and should remain unparsed. For any other attribute you better have a very good reason
/// not to parse it here.
pub fn parse_attribute<'a>(
    dcx: DiagCtxtHandle<'_>,
    attr: &'a ast::Attribute,
) -> Result<MaybeParsedAttribute<'a>, ErrorGuaranteed> {
    let res = match &attr.kind {
        ast::AttrKind::DocComment(comment_kind, symbol) => {
            MaybeParsedAttribute::Parsed(ParsedAttributeKind::DocComment(*comment_kind, *symbol))
        }
        // FIXME(jdonszelmann): check whether n is a registered tool to reduce mistakes
        ast::AttrKind::Normal(n) => MaybeParsedAttribute::MustRemainUnparsed(&*n),
    };

    Ok(res)
}

pub trait AttributeCollection<'a> {
    fn iter(self) -> impl Iterator<Item = &'a Attribute> + 'a;
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

impl<'a> AttributeCollection<'a> for &'a [Attribute] {
    fn iter(self) -> impl Iterator<Item = &'a Attribute> + 'a {
        self.iter()
    }
}
impl<'a> AttributeCollection<'a> for &'a Vec<Attribute> {
    fn iter(self) -> impl Iterator<Item = &'a Attribute> + 'a {
        <&[_]>::iter(self)
    }
}

pub struct AttributeIterator<'a, I>(pub I, PhantomData<&'a ()>);
impl<'a, I> AttributeIterator<'a, I> {
    pub fn new(i: I) -> Self {
        Self(i, PhantomData)
    }
}

impl<'a, I: Iterator<Item = &'a Attribute>> Iterator for AttributeIterator<'a, I> {
    type Item = &'a Attribute;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, I: Iterator<Item = &'a Attribute> + 'a> AttributeCollection<'a>
    for AttributeIterator<'a, I>
{
    fn iter(self) -> impl Iterator<Item = &'a Attribute> + 'a {
        self
    }
}
