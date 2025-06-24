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
//! Attributes should be added to `crate::context::ATTRIBUTE_PARSERS` to be parsed.

use std::marker::PhantomData;

use rustc_attr_data_structures::AttributeKind;
use rustc_feature::AttributeTemplate;
use rustc_span::{Span, Symbol};
use thin_vec::ThinVec;

use crate::context::{AcceptContext, FinalizeContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics::UnusedMultiple;

pub(crate) mod allow_unstable;
pub(crate) mod cfg;
pub(crate) mod codegen_attrs;
pub(crate) mod confusables;
pub(crate) mod deprecation;
pub(crate) mod inline;
pub(crate) mod lint_helpers;
pub(crate) mod must_use;
pub(crate) mod repr;
pub(crate) mod semantics;
pub(crate) mod stability;
pub(crate) mod traits;
pub(crate) mod transparency;
pub(crate) mod util;

type AcceptFn<T, S> = for<'sess> fn(&mut T, &mut AcceptContext<'_, 'sess, S>, &ArgParser<'_>);
type AcceptMapping<T, S> = &'static [(&'static [Symbol], AttributeTemplate, AcceptFn<T, S>)];

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
///
/// For a simpler attribute parsing interface, consider using [`SingleAttributeParser`]
/// or [`CombineAttributeParser`] instead.
pub(crate) trait AttributeParser<S: Stage>: Default + 'static {
    /// The symbols for the attributes that this parser is interested in.
    ///
    /// If an attribute has this symbol, the `accept` function will be called on it.
    const ATTRIBUTES: AcceptMapping<Self, S>;

    /// The parser has gotten a chance to accept the attributes on an item,
    /// here it can produce an attribute.
    ///
    /// All finalize methods of all parsers are unconditionally called.
    /// This means you can't unconditionally return `Some` here,
    /// that'd be equivalent to unconditionally applying an attribute to
    /// every single syntax item that could have attributes applied to it.
    /// Your accept mappings should determine whether this returns something.
    fn finalize(self, cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind>;
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
pub(crate) trait SingleAttributeParser<S: Stage>: 'static {
    /// The single path of the attribute this parser accepts.
    ///
    /// If you need the parser to accept more than one path, use [`AttributeParser`] instead
    const PATH: &[Symbol];

    /// Configures the precedence of attributes with the same `PATH` on a syntax node.
    const ATTRIBUTE_ORDER: AttributeOrder;

    /// Configures what to do when when the same attribute is
    /// applied more than once on the same syntax node.
    ///
    /// [`ATTRIBUTE_ORDER`](Self::ATTRIBUTE_ORDER) specified which one is assumed to be correct,
    /// and this specified whether to, for example, warn or error on the other one.
    const ON_DUPLICATE: OnDuplicate<S>;

    /// The template this attribute parser should implement. Used for diagnostics.
    const TEMPLATE: AttributeTemplate;

    /// Converts a single syntactical attribute to a single semantic attribute, or [`AttributeKind`]
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind>;
}

/// Use in combination with [`SingleAttributeParser`].
/// `Single<T: SingleAttributeParser>` implements [`AttributeParser`].
pub(crate) struct Single<T: SingleAttributeParser<S>, S: Stage>(
    PhantomData<(S, T)>,
    Option<(AttributeKind, Span)>,
);

impl<T: SingleAttributeParser<S>, S: Stage> Default for Single<T, S> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: SingleAttributeParser<S>, S: Stage> AttributeParser<S> for Single<T, S> {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        T::PATH,
        <T as SingleAttributeParser<S>>::TEMPLATE,
        |group: &mut Single<T, S>, cx, args| {
            if let Some(pa) = T::convert(cx, args) {
                match T::ATTRIBUTE_ORDER {
                    // keep the first and report immediately. ignore this attribute
                    AttributeOrder::KeepFirst => {
                        if let Some((_, unused)) = group.1 {
                            T::ON_DUPLICATE.exec::<T>(cx, cx.attr_span, unused);
                            return;
                        }
                    }
                    // keep the new one and warn about the previous,
                    // then replace
                    AttributeOrder::KeepLast => {
                        if let Some((_, used)) = group.1 {
                            T::ON_DUPLICATE.exec::<T>(cx, used, cx.attr_span);
                        }
                    }
                }

                group.1 = Some((pa, cx.attr_span));
            }
        },
    )];

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        Some(self.1?.0)
    }
}

// FIXME(jdonszelmann): logic is implemented but the attribute parsers needing
// them will be merged in another PR
#[allow(unused)]
pub(crate) enum OnDuplicate<S: Stage> {
    /// Give a default warning
    Warn,

    /// Duplicates will be a warning, with a note that this will be an error in the future.
    WarnButFutureError,

    /// Give a default error
    Error,

    /// Ignore duplicates
    Ignore,

    /// Custom function called when a duplicate attribute is found.
    ///
    /// - `unused` is the span of the attribute that was unused or bad because of some
    ///   duplicate reason (see [`AttributeOrder`])
    /// - `used` is the span of the attribute that was used in favor of the unused attribute
    Custom(fn(cx: &AcceptContext<'_, '_, S>, used: Span, unused: Span)),
}

impl<S: Stage> OnDuplicate<S> {
    fn exec<P: SingleAttributeParser<S>>(
        &self,
        cx: &mut AcceptContext<'_, '_, S>,
        used: Span,
        unused: Span,
    ) {
        match self {
            OnDuplicate::Warn => cx.warn_unused_duplicate(used, unused),
            OnDuplicate::WarnButFutureError => cx.warn_unused_duplicate_future_error(used, unused),
            OnDuplicate::Error => {
                cx.emit_err(UnusedMultiple {
                    this: used,
                    other: unused,
                    name: Symbol::intern(
                        &P::PATH.into_iter().map(|i| i.to_string()).collect::<Vec<_>>().join(".."),
                    ),
                });
            }
            OnDuplicate::Ignore => {}
            OnDuplicate::Custom(f) => f(cx, used, unused),
        }
    }
}
//
// FIXME(jdonszelmann): logic is implemented but the attribute parsers needing
// them will be merged in another PR
#[allow(unused)]
pub(crate) enum AttributeOrder {
    /// Duplicates after the first attribute will be an error.
    ///
    /// This should be used where duplicates would be ignored, but carry extra
    /// meaning that could cause confusion. For example, `#[stable(since="1.0")]
    /// #[stable(since="2.0")]`, which version should be used for `stable`?
    KeepFirst,

    /// Duplicates preceding the last instance of the attribute will be a
    /// warning, with a note that this will be an error in the future.
    ///
    /// This is the same as `FutureWarnFollowing`, except the last attribute is
    /// the one that is "used". Ideally these can eventually migrate to
    /// `ErrorPreceding`.
    KeepLast,
}

type ConvertFn<E> = fn(ThinVec<E>) -> AttributeKind;

/// Alternative to [`AttributeParser`] that automatically handles state management.
/// If multiple attributes appear on an element, combines the values of each into a
/// [`ThinVec`].
/// [`Combine<T> where T: CombineAttributeParser`](Combine) implements [`AttributeParser`].
///
/// [`CombineAttributeParser`] can only convert a single kind of attribute, and cannot combine multiple
/// attributes together like is necessary for `#[stable()]` and `#[unstable()]` for example.
pub(crate) trait CombineAttributeParser<S: Stage>: 'static {
    const PATH: &[rustc_span::Symbol];

    type Item;
    /// A function that converts individual items (of type [`Item`](Self::Item)) into the final attribute.
    ///
    /// For example, individual representations fomr `#[repr(...)]` attributes into an `AttributeKind::Repr(x)`,
    ///  where `x` is a vec of these individual reprs.
    const CONVERT: ConvertFn<Self::Item>;

    /// The template this attribute parser should implement. Used for diagnostics.
    const TEMPLATE: AttributeTemplate;

    /// Converts a single syntactical attribute to a number of elements of the semantic attribute, or [`AttributeKind`]
    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c;
}

/// Use in combination with [`CombineAttributeParser`].
/// `Combine<T: CombineAttributeParser>` implements [`AttributeParser`].
pub(crate) struct Combine<T: CombineAttributeParser<S>, S: Stage>(
    PhantomData<(S, T)>,
    ThinVec<<T as CombineAttributeParser<S>>::Item>,
);

impl<T: CombineAttributeParser<S>, S: Stage> Default for Combine<T, S> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: CombineAttributeParser<S>, S: Stage> AttributeParser<S> for Combine<T, S> {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        T::PATH,
        <T as CombineAttributeParser<S>>::TEMPLATE,
        |group: &mut Combine<T, S>, cx, args| group.1.extend(T::extend(cx, args)),
    )];

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if self.1.is_empty() { None } else { Some(T::CONVERT(self.1)) }
    }
}
