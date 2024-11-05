//! [`AttributeGroup`]s are groups of attributes (groups can be size 1) that are parsed together.
//! An [`AttributeGroup`] implementation defines its parser.
//!
//! You can find more docs on what groups are on [`AttributeGroup`] itself.
//! However, for many types of attributes, implementing [`AttributeGroup`] is not necessary.
//! It allows for a lot of flexibility you might not want.
//!
//! Specifically, you care about managing the state of your [AttributeGroup] state machine
//! yourself. In this case you can choose to implement:
//!
//! - [`SingleAttributeGroup`]: makes it easy to implement an attribute which should error if it
//! appears more than once in a list of attributes
//! - [`CombineAttributeGroup`]: makes it easy to implement an attribute which should combine the
//! contents of attributes, if an attribute appear multiple times in a list
//!
//! Attributes should be added to [`ATTRIBUTE_GROUP_MAPPING`](crate::context::ATTRIBUTE_GROUP_MAPPING) to be parsed.

use std::marker::PhantomData;

use rustc_ast::Expr;
use rustc_hir::AttributeKind;
use rustc_span::{ErrorGuaranteed, Span};
use thin_vec::ThinVec;

use crate::context::{AttributeAcceptContext, AttributeGroupContext};
use crate::parser::GenericArgParser;

pub(crate) mod allow_unstable;
pub(crate) mod cfg;
pub(crate) mod confusables;
pub(crate) mod deprecation;
pub(crate) mod inline;
pub(crate) mod repr;
pub(crate) mod stability;
pub(crate) mod transparency;
pub(crate) mod util;

type AttributeHandler<T> = fn(&mut T, &AttributeAcceptContext<'_>, &GenericArgParser<'_, Expr>);
type AttributeMapping<T> = &'static [(&'static [rustc_span::Symbol], AttributeHandler<T>)];

/// An [`AttributeGroup`] is a type which searches for syntactic attributes.
///
/// Groups are often tiny state machines. [`Default::default`]
/// creates a new instance that sits in some kind of initial state, usually that the
/// attribute it is looking for was not yet seen.
///
/// Then, it defines what paths this group will accept in [`AttributeGroup::ATTRIBUTES`].
/// These are listed as pairs, of symbols and function pointers. The function pointer will
/// be called when that attribute is found on an item, which can influence the state.
///
/// Finally, all `finalize` functions are called, for each piece of state,
pub(crate) trait AttributeGroup: Default + 'static {
    /// The symbols for the attributes that this extractor can extract.
    ///
    /// If an attribute has this symbol, the `accept` function will be called on it.
    const ATTRIBUTES: AttributeMapping<Self>;

    /// The extractor has gotten a chance to accept the attributes on an item,
    /// now produce an attribute.
    fn finalize(self, cx: &AttributeGroupContext<'_>) -> Option<(AttributeKind, AttributeFilter)>;
}

/// Create an AttributeFilter using [`attribute_filter`].
///
/// Tells the parsing system what other attributes this attribute can be used with together.
pub(crate) struct AttributeFilter(Box<dyn Fn(&AttributeKind) -> Result<(), ErrorGuaranteed>>);

/// A slightly simpler and more restricted way to convert attributes which you can implement for
/// unit types. Assumes that a single attribute can only appear a single time on an item
/// [`SingleGroup<T> where T: SingleAttributeGroup`](Single) creates an [`AttributeGroup`] from any [`SingleAttributeGroup`].
///
/// [`SingleGroup`] can only convert attributes one-to-one, and cannot combine multiple
/// attributes together like is necessary for `#[stable()]` and `#[unstable()]` for example.
pub(crate) trait SingleAttributeGroup: 'static {
    const PATH: &'static [rustc_span::Symbol];

    /// Caled when a duplicate attribute is found.
    ///
    /// `first_span` is the span of the first occurrence of this attribute.
    fn on_duplicate(cx: &AttributeAcceptContext<'_>, first_span: Span);

    /// The extractor has gotten a chance to accept the attributes on an item,
    /// now produce an attribute.
    fn convert(
        cx: &AttributeAcceptContext<'_>,
        args: &GenericArgParser<'_, Expr>,
    ) -> Option<(AttributeKind, AttributeFilter)>;
}

pub(crate) struct Single<T: SingleAttributeGroup>(
    PhantomData<T>,
    Option<(AttributeKind, AttributeFilter, Span)>,
);

impl<T: SingleAttributeGroup> Default for Single<T> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: SingleAttributeGroup> AttributeGroup for Single<T> {
    const ATTRIBUTES: AttributeMapping<Self> = &[(T::PATH, |group: &mut Single<T>, cx, args| {
        if let Some((_, _, s)) = group.1 {
            T::on_duplicate(cx, s);
            return;
        }

        if let Some((pa, f)) = T::convert(cx, args) {
            group.1 = Some((pa, f, cx.attr_span));
        }
    })];

    fn finalize(self, _cx: &AttributeGroupContext<'_>) -> Option<(AttributeKind, AttributeFilter)> {
        let (pa, f, _) = self.1?;
        Some((pa, f))
    }
}

type ConvertFn<E> = fn(ThinVec<E>) -> AttributeKind;

/// A slightly simpler and more restricted way to convert attributes which you can implement for
/// unit types. If multiple attributes appear on an element, combines the values of each into a
/// [`ThinVec`].
/// [`CombineGroup<T> where T: CombineAttributeGroup`](Combine) creates an [`AttributeGroup`] from any [`CombineAttributeGroup`].
///
/// [`CombineAttributeGroup`] can only convert a single kind of attribute, and cannot combine multiple
/// attributes together like is necessary for `#[stable()]` and `#[unstable()]` for example.
pub(crate) trait CombineAttributeGroup: 'static {
    const PATH: &'static [rustc_span::Symbol];

    type Item;
    const CONVERT: ConvertFn<Self::Item>;

    /// The extractor has gotten a chance to accept the attributes on an item,
    /// now produce an attribute.
    fn extend<'a>(
        cx: &'a AttributeAcceptContext<'a>,
        args: &'a GenericArgParser<'a, Expr>,
    ) -> impl IntoIterator<Item = Self::Item> + 'a;
}

pub(crate) struct Combine<T: CombineAttributeGroup>(
    PhantomData<T>,
    ThinVec<<T as CombineAttributeGroup>::Item>,
);

impl<T: CombineAttributeGroup> Default for Combine<T> {
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T: CombineAttributeGroup> AttributeGroup for Combine<T> {
    const ATTRIBUTES: AttributeMapping<Self> =
        &[(T::PATH, |group: &mut Combine<T>, cx, args| group.1.extend(T::extend(cx, args)))];

    fn finalize(self, _cx: &AttributeGroupContext<'_>) -> Option<(AttributeKind, AttributeFilter)> {
        if self.1.is_empty() {
            None
        } else {
            // TODO: what filter here?
            Some((T::CONVERT(self.1), crate::attribute_filter!(allow all)))
        }
    }
}

#[macro_export]
macro_rules! attribute_filter {
    (deny all: $b: block) => {
        $crate::attribute_filter!(allow: else $b)
    };

    (allow all) => {
        $crate::attribute_filter!(deny: )
    };

    (
        allow:
        $(
            [$pat: pat $(if $expr: expr)?]
        ),*
        else $b: block
    ) => {
        $crate::attributes::AttributeFilter(Box::new(|attr: &rustc_hir::AttributeKind| -> Result<(), rustc_span::ErrorGuaranteed> {
            match attr {
                $(
                    $pat $(if $expr)? => Ok(()),
                )*
                _ => Err($b: block),
            }
        }))
    };

    (deny: $(
        [$pat: pat $(if $expr: expr)? $(=> $b: block)?]
    ),*) => {
        $crate::attributes::AttributeFilter(Box::new(|attr: &rustc_hir::AttributeKind| -> Result<(), rustc_span::ErrorGuaranteed> {
            match attr {
                $(
                    $pat $(if $expr)? => Err($b),
                )*
                _ => Ok(()),
            }
        }))
    };
}
