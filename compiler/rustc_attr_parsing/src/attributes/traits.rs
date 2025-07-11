use core::mem;

use rustc_attr_data_structures::AttributeKind;
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{
    AttributeOrder, NoArgsAttributeParser, OnDuplicate, SingleAttributeParser,
};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;

pub(crate) struct SkipDuringMethodDispatchParser;
impl<S: Stage> SingleAttributeParser<S> for SkipDuringMethodDispatchParser {
    const PATH: &[Symbol] = &[sym::rustc_skip_during_method_dispatch];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;

    const TEMPLATE: AttributeTemplate = template!(List: "array, boxed_slice");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let mut array = false;
        let mut boxed_slice = false;
        let Some(args) = args.list() else {
            cx.expected_list(cx.attr_span);
            return None;
        };
        if args.is_empty() {
            cx.expected_at_least_one_argument(args.span);
            return None;
        }
        for arg in args.mixed() {
            let Some(arg) = arg.meta_item() else {
                cx.unexpected_literal(arg.span());
                continue;
            };
            if let Err(span) = arg.args().no_args() {
                cx.expected_no_args(span);
            }
            let path = arg.path();
            let (key, skip): (Symbol, &mut bool) = match path.word_sym() {
                Some(key @ sym::array) => (key, &mut array),
                Some(key @ sym::boxed_slice) => (key, &mut boxed_slice),
                _ => {
                    cx.expected_specific_argument(path.span(), vec!["array", "boxed_slice"]);
                    continue;
                }
            };
            if mem::replace(skip, true) {
                cx.duplicate_key(arg.span(), key);
            }
        }
        Some(AttributeKind::SkipDuringMethodDispatch { array, boxed_slice, span: cx.attr_span })
    }
}

pub(crate) struct ParenSugarParser;
impl<S: Stage> NoArgsAttributeParser<S> for ParenSugarParser {
    const PATH: &[Symbol] = &[sym::rustc_paren_sugar];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ParenSugar;
}

pub(crate) struct TypeConstParser;
impl<S: Stage> NoArgsAttributeParser<S> for TypeConstParser {
    const PATH: &[Symbol] = &[sym::type_const];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::TypeConst;
}

// Markers

pub(crate) struct MarkerParser;
impl<S: Stage> NoArgsAttributeParser<S> for MarkerParser {
    const PATH: &[Symbol] = &[sym::marker];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Marker;
}

pub(crate) struct DenyExplicitImplParser;
impl<S: Stage> NoArgsAttributeParser<S> for DenyExplicitImplParser {
    const PATH: &[Symbol] = &[sym::rustc_deny_explicit_impl];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::DenyExplicitImpl;
}

pub(crate) struct DoNotImplementViaObjectParser;
impl<S: Stage> NoArgsAttributeParser<S> for DoNotImplementViaObjectParser {
    const PATH: &[Symbol] = &[sym::rustc_do_not_implement_via_object];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::DoNotImplementViaObject;
}

// Const traits

pub(crate) struct ConstTraitParser;
impl<S: Stage> NoArgsAttributeParser<S> for ConstTraitParser {
    const PATH: &[Symbol] = &[sym::const_trait];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ConstTrait;
}

// Specialization

pub(crate) struct SpecializationTraitParser;
impl<S: Stage> NoArgsAttributeParser<S> for SpecializationTraitParser {
    const PATH: &[Symbol] = &[sym::rustc_specialization_trait];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::SpecializationTrait;
}

pub(crate) struct UnsafeSpecializationMarkerParser;
impl<S: Stage> NoArgsAttributeParser<S> for UnsafeSpecializationMarkerParser {
    const PATH: &[Symbol] = &[sym::rustc_unsafe_specialization_marker];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::UnsafeSpecializationMarker;
}

// Coherence

pub(crate) struct CoinductiveParser;
impl<S: Stage> NoArgsAttributeParser<S> for CoinductiveParser {
    const PATH: &[Symbol] = &[sym::rustc_coinductive];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Coinductive;
}

pub(crate) struct AllowIncoherentImplParser;
impl<S: Stage> NoArgsAttributeParser<S> for AllowIncoherentImplParser {
    const PATH: &[Symbol] = &[sym::rustc_allow_incoherent_impl];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::AllowIncoherentImpl;
}

pub(crate) struct CoherenceIsCoreParser;
impl<S: Stage> NoArgsAttributeParser<S> for CoherenceIsCoreParser {
    const PATH: &[Symbol] = &[sym::rustc_coherence_is_core];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::CoherenceIsCore;
}

pub(crate) struct FundamentalParser;
impl<S: Stage> NoArgsAttributeParser<S> for FundamentalParser {
    const PATH: &[Symbol] = &[sym::fundamental];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::Fundamental;
}
