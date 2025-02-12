//! This module defines traits for attribute parsers, little state machines that recognize and parse
//! attributes out of a longer list of attributes. The main trait is called [`AttributeParser`].
//! You can find more docs about [`AttributeParser`]s on the trait itself.
//! However, for many types of attributes, implementing [`AttributeParser`] is not necessary.
//! It allows for a lot of flexibility you might not want.
//!
//! Specifically, you might not care about managing the state of your [`AttributeParser`]
//! state machine yourself. In this case you can choose to implement:
//!
//! - [`SingleAttributeParser`]: makes it easy to implement an attribute which should error if it
//! appears more than once in a list of attributes
//! - [`CombineAttributeParser`]: makes it easy to implement an attribute which should combine the
//! contents of attributes, if an attribute appear multiple times in a list
//!
//! Attributes should be added to [`ATTRIBUTE_MAPPING`](crate::context::ATTRIBUTE_MAPPING) to be parsed.

use std::marker::PhantomData;

use rustc_attr_data_structures::AttributeKind;
use rustc_span::Span;
use thin_vec::ThinVec;

use crate::context::{AcceptContext, FinalizeContext};
use crate::parser::ArgParser;

pub(crate) mod allow_unstable;
pub(crate) mod cfg;
pub(crate) mod confusables;
pub(crate) mod deprecation;
pub(crate) mod repr;
pub(crate) mod stability;
pub(crate) mod transparency;
pub(crate) mod util;

type AcceptFn<T> = fn(&mut T, &AcceptContext<'_>, &ArgParser<'_>);
type AcceptMapping<T> = &'static [(&'static [rustc_span::Symbol], AcceptFn<T>)];

/// An [`AttributeParser`] is a type which searches for syntactic attributes.
///
/// Parsers are often tiny state machines that gets to see all syntactical attributes on an item.
/// [`Default::default`] creates a fresh instance that sits in some kind of initial state, usually that the
/// attribute it is looking for was not yet seen.
///
/// Then, it defines what paths this group will accept in [`AttributeParser::ATTRIBUTES`].
/// These are listed as pairs, of symbols and function pointers. The function pointer will
/// be called when that attribute is found on an item, which can influence the state of the little
/// state machine.
///
/// Finally, after all attributes on an item have been seen, and possibly been accepted,
/// the [`finalize`](AttributeParser::finalize) functions for all attribute parsers are called. Each can then report
/// whether it has seen the attribute it has been looking for.
///
/// The state machine is automatically reset to parse attributes on the next item.
pub(crate) trait AttributeParser: Default + 'static {
    /// The symbols for the attributes that this parser is interested in.
    ///
    /// If an attribute has this symbol, the `accept` function will be called on it.
    const ATTRIBUTES: AcceptMapping<Self>;

    /// The parser has gotten a chance to accept the attributes on an item,
    /// here it can produce an attribute.
    fn finalize(self, cx: &FinalizeContext<'_>) -> Option<AttributeKind>;
}

/// Alternative to [`AttributeParser`] that automatically handles state management.
/// A slightly simpler and more restricted way to convert attributes.
/// Assumes that an attribute can only appear a single time on an item,
/// and errors when it sees more.
///
/// [`Single<T> where T: SingleAttributeParser`](Single) implements [`AttributeParser`].
///
/// [`SingleAttributeParser`] can only convert attributes one-to-one, and cannot combine multiple
/// attributes together like is necessary for `#[stable()]` and `#[unstable()]` for example.
pub(crate) trait SingleAttributeParser: 'static {
    const PATH: &'static [rustc_span::Symbol];

    /// Caled when a duplicate attribute is found.
    ///
    /// `first_span` is the span of the first occurrence of this attribute.
    // FIXME(jdonszelmann): default error
    fn on_duplicate(cx: &AcceptContext<'_>, first_span: Span);

    /// Converts a single syntactical attribute to a single semantic attribute, or [`AttributeKind`]
    fn convert(cx: &AcceptContext<'_>, args: &ArgParser<'_>) -> Option<AttributeKind>;
}

pub(crate) struct Single<T: SingleAttributeParser>(PhantomData<T>, Option<(AttributeKind, Span)>);

impl<T: SingleAttributeParser> Default for Single<T> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: SingleAttributeParser> AttributeParser for Single<T> {
    const ATTRIBUTES: AcceptMapping<Self> = &[(T::PATH, |group: &mut Single<T>, cx, args| {
        if let Some((_, s)) = group.1 {
            T::on_duplicate(cx, s);
            return;
        }

        if let Some(pa) = T::convert(cx, args) {
            group.1 = Some((pa, cx.attr_span));
        }
    })];

    fn finalize(self, _cx: &FinalizeContext<'_>) -> Option<AttributeKind> {
        Some(self.1?.0)
    }
}

type ConvertFn<E> = fn(ThinVec<E>) -> AttributeKind;

/// Alternative to [`AttributeParser`] that automatically handles state management.
/// If multiple attributes appear on an element, combines the values of each into a
/// [`ThinVec`].
/// [`Combine<T> where T: CombineAttributeParser`](Combine) implements [`AttributeParser`].
///
/// [`CombineAttributeParser`] can only convert a single kind of attribute, and cannot combine multiple
/// attributes together like is necessary for `#[stable()]` and `#[unstable()]` for example.
pub(crate) trait CombineAttributeParser: 'static {
    const PATH: &'static [rustc_span::Symbol];

    type Item;
    const CONVERT: ConvertFn<Self::Item>;

    /// Converts a single syntactical attribute to a number of elements of the semantic attribute, or [`AttributeKind`]
    fn extend<'a>(
        cx: &'a AcceptContext<'a>,
        args: &'a ArgParser<'a>,
    ) -> impl IntoIterator<Item = Self::Item> + 'a;
}

pub(crate) struct Combine<T: CombineAttributeParser>(
    PhantomData<T>,
    ThinVec<<T as CombineAttributeParser>::Item>,
);

impl<T: CombineAttributeParser> Default for Combine<T> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: CombineAttributeParser> AttributeParser for Combine<T> {
    const ATTRIBUTES: AcceptMapping<Self> =
        &[(T::PATH, |group: &mut Combine<T>, cx, args| group.1.extend(T::extend(cx, args)))];

    fn finalize(self, _cx: &FinalizeContext<'_>) -> Option<AttributeKind> {
        if self.1.is_empty() { None } else { Some(T::CONVERT(self.1)) }
    }
}
