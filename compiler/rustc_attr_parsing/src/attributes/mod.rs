//! This module defines traits for attribute parsers, little state machines that recognize and parse
//! attributes out of a longer list of attributes. The main trait is called [`AttributeParser`].
//! You can find more docs about [`AttributeParser`]s on the trait itself.
//! However, for many types of attributes, implementing [`AttributeParser`] is not necessary.
//! It allows for a lot of flexibility you might not want.
//!
//! Specifically, you might not care about managing the state of your [`AttributeParser`]
//! state machine yourself. In this case you can choose to implement:
//!
//! - [`SingleAttributeParser`](crate::attributes::SingleAttributeParser): makes it easy to implement an attribute which should error if it
//! appears more than once in a list of attributes
//! - [`CombineAttributeParser`](crate::attributes::CombineAttributeParser): makes it easy to implement an attribute which should combine the
//! contents of attributes, if an attribute appear multiple times in a list
//!
//! Attributes should be added to `crate::context::ATTRIBUTE_PARSERS` to be parsed.

use std::marker::PhantomData;

use rustc_feature::{AttributeTemplate, template};
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol};
use thin_vec::ThinVec;

use crate::context::{AcceptContext, FinalizeContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics::UnusedMultiple;
use crate::target_checking::AllowedTargets;

/// All the parsers require roughly the same imports, so this prelude has most of the often-needed ones.
mod prelude;

pub(crate) mod allow_unstable;
pub(crate) mod body;
pub(crate) mod cfg;
pub(crate) mod cfg_old;
pub(crate) mod codegen_attrs;
pub(crate) mod confusables;
pub(crate) mod crate_level;
pub(crate) mod debugger;
pub(crate) mod deprecation;
pub(crate) mod dummy;
pub(crate) mod inline;
pub(crate) mod link_attrs;
pub(crate) mod lint_helpers;
pub(crate) mod loop_match;
pub(crate) mod macro_attrs;
pub(crate) mod must_use;
pub(crate) mod no_implicit_prelude;
pub(crate) mod non_exhaustive;
pub(crate) mod path;
pub(crate) mod proc_macro_attrs;
pub(crate) mod prototype;
pub(crate) mod repr;
pub(crate) mod rustc_internal;
pub(crate) mod semantics;
pub(crate) mod stability;
pub(crate) mod test_attrs;
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
    const ALLOWED_TARGETS: AllowedTargets;

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

    const ALLOWED_TARGETS: AllowedTargets;

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
                    AttributeOrder::KeepInnermost => {
                        if let Some((_, unused)) = group.1 {
                            T::ON_DUPLICATE.exec::<T>(cx, cx.attr_span, unused);
                            return;
                        }
                    }
                    // keep the new one and warn about the previous,
                    // then replace
                    AttributeOrder::KeepOutermost => {
                        if let Some((_, used)) = group.1 {
                            T::ON_DUPLICATE.exec::<T>(cx, used, cx.attr_span);
                        }
                    }
                }

                group.1 = Some((pa, cx.attr_span));
            }
        },
    )];
    const ALLOWED_TARGETS: AllowedTargets = T::ALLOWED_TARGETS;

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        Some(self.1?.0)
    }
}

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

pub(crate) enum AttributeOrder {
    /// Duplicates after the innermost instance of the attribute will be an error/warning.
    /// Only keep the lowest attribute.
    ///
    /// Attributes are processed from bottom to top, so this raises a warning/error on all the attributes
    /// further above the lowest one:
    /// ```
    /// #[stable(since="1.0")] //~ WARNING duplicated attribute
    /// #[stable(since="2.0")]
    /// ```
    KeepInnermost,

    /// Duplicates before the outermost instance of the attribute will be an error/warning.
    /// Only keep the highest attribute.
    ///
    /// Attributes are processed from bottom to top, so this raises a warning/error on all the attributes
    /// below the highest one:
    /// ```
    /// #[path="foo.rs"]
    /// #[path="bar.rs"] //~ WARNING duplicated attribute
    /// ```
    KeepOutermost,
}

/// An even simpler version of [`SingleAttributeParser`]:
/// now automatically check that there are no arguments provided to the attribute.
///
/// [`WithoutArgs<T> where T: NoArgsAttributeParser`](WithoutArgs) implements [`SingleAttributeParser`].
//
pub(crate) trait NoArgsAttributeParser<S: Stage>: 'static {
    const PATH: &[Symbol];
    const ON_DUPLICATE: OnDuplicate<S>;
    const ALLOWED_TARGETS: AllowedTargets;

    /// Create the [`AttributeKind`] given attribute's [`Span`].
    const CREATE: fn(Span) -> AttributeKind;
}

pub(crate) struct WithoutArgs<T: NoArgsAttributeParser<S>, S: Stage>(PhantomData<(S, T)>);

impl<T: NoArgsAttributeParser<S>, S: Stage> Default for WithoutArgs<T, S> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: NoArgsAttributeParser<S>, S: Stage> SingleAttributeParser<S> for WithoutArgs<T, S> {
    const PATH: &[Symbol] = T::PATH;
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = T::ON_DUPLICATE;
    const ALLOWED_TARGETS: AllowedTargets = T::ALLOWED_TARGETS;
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
        }
        Some(T::CREATE(cx.attr_span))
    }
}

type ConvertFn<E> = fn(ThinVec<E>, Span) -> AttributeKind;

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

    const ALLOWED_TARGETS: AllowedTargets;

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
pub(crate) struct Combine<T: CombineAttributeParser<S>, S: Stage> {
    phantom: PhantomData<(S, T)>,
    /// A list of all items produced by parsing attributes so far. One attribute can produce any amount of items.
    items: ThinVec<<T as CombineAttributeParser<S>>::Item>,
    /// The full span of the first attribute that was encountered.
    first_span: Option<Span>,
}

impl<T: CombineAttributeParser<S>, S: Stage> Default for Combine<T, S> {
    fn default() -> Self {
        Self {
            phantom: Default::default(),
            items: Default::default(),
            first_span: Default::default(),
        }
    }
}

impl<T: CombineAttributeParser<S>, S: Stage> AttributeParser<S> for Combine<T, S> {
    const ATTRIBUTES: AcceptMapping<Self, S> =
        &[(T::PATH, T::TEMPLATE, |group: &mut Combine<T, S>, cx, args| {
            // Keep track of the span of the first attribute, for diagnostics
            group.first_span.get_or_insert(cx.attr_span);
            group.items.extend(T::extend(cx, args))
        })];
    const ALLOWED_TARGETS: AllowedTargets = T::ALLOWED_TARGETS;

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        if let Some(first_span) = self.first_span {
            Some(T::CONVERT(self.items, first_span))
        } else {
            None
        }
    }
}
