use std::mem;

use super::prelude::*;
use crate::attributes::{NoArgsAttributeParser, OnDuplicate, SingleAttributeParser};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::{Allow, Warn};

pub(crate) struct RustcSkipDuringMethodDispatchParser;
impl<S: Stage> SingleAttributeParser<S> for RustcSkipDuringMethodDispatchParser {
    const PATH: &[Symbol] = &[sym::rustc_skip_during_method_dispatch];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_skip_during_method_dispatch,
        "the `#[rustc_skip_during_method_dispatch]` attribute is used to exclude a trait \
        from method dispatch when the receiver is of the following type, for compatibility in \
        editions < 2021 (array) or editions < 2024 (boxed_slice)"
    );
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);

    const TEMPLATE: AttributeTemplate = template!(List: &["array, boxed_slice"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let mut array = false;
        let mut boxed_slice = false;
        let Some(args) = args.list() else {
            let attr_span = cx.attr_span;
            cx.adcx().expected_list(attr_span, args);
            return None;
        };
        if args.is_empty() {
            cx.adcx().expected_at_least_one_argument(args.span);
            return None;
        }
        for arg in args.mixed() {
            let Some(arg) = arg.meta_item() else {
                cx.adcx().expected_not_literal(arg.span());
                continue;
            };
            if let Err(span) = arg.args().no_args() {
                cx.adcx().expected_no_args(span);
            }
            let path = arg.path();
            let (key, skip): (Symbol, &mut bool) = match path.word_sym() {
                Some(key @ sym::array) => (key, &mut array),
                Some(key @ sym::boxed_slice) => (key, &mut boxed_slice),
                _ => {
                    cx.adcx()
                        .expected_specific_argument(path.span(), &[sym::array, sym::boxed_slice]);
                    continue;
                }
            };
            if mem::replace(skip, true) {
                cx.adcx().duplicate_key(arg.span(), key);
            }
        }
        Some(AttributeKind::RustcSkipDuringMethodDispatch {
            array,
            boxed_slice,
            span: cx.attr_span,
        })
    }
}

pub(crate) struct RustcParenSugarParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcParenSugarParser {
    const PATH: &[Symbol] = &[sym::rustc_paren_sugar];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated!(unboxed_closures, "unboxed_closures are still evolving");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcParenSugar;
}

// Markers

pub(crate) struct MarkerParser;
impl<S: Stage> NoArgsAttributeParser<S> for MarkerParser {
    const PATH: &[Symbol] = &[sym::marker];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const GATED: AttributeGate = gated!(marker_trait_attr, experimental!(marker));
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Trait),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Marker;
}

pub(crate) struct RustcDenyExplicitImplParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcDenyExplicitImplParser {
    const PATH: &[Symbol] = &[sym::rustc_deny_explicit_impl];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_deny_explicit_impl,
        "`#[rustc_deny_explicit_impl]` enforces that a trait can have no user-provided impls"
    );
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcDenyExplicitImpl;
}

pub(crate) struct RustcDynIncompatibleTraitParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcDynIncompatibleTraitParser {
    const PATH: &[Symbol] = &[sym::rustc_dyn_incompatible_trait];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_dyn_incompatible_trait,
        "`#[rustc_dyn_incompatible_trait]` marks a trait as dyn-incompatible, \
        even if it otherwise satisfies the requirements to be dyn-compatible."
    );
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcDynIncompatibleTrait;
}

// Specialization

pub(crate) struct RustcSpecializationTraitParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcSpecializationTraitParser {
    const PATH: &[Symbol] = &[sym::rustc_specialization_trait];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_specialization_trait,
        "the `#[rustc_specialization_trait]` attribute is used to check specializations"
    );
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcSpecializationTrait;
}

pub(crate) struct RustcUnsafeSpecializationMarkerParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcUnsafeSpecializationMarkerParser {
    const PATH: &[Symbol] = &[sym::rustc_unsafe_specialization_marker];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_unsafe_specialization_marker,
        "the `#[rustc_unsafe_specialization_marker]` attribute is used to check specializations"
    );
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcUnsafeSpecializationMarker;
}

// Coherence

pub(crate) struct RustcCoinductiveParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcCoinductiveParser {
    const PATH: &[Symbol] = &[sym::rustc_coinductive];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_coinductive,
        "`#[rustc_coinductive]` changes a trait to be coinductive, allowing cycles in the trait solver"
    );

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcCoinductive;
}

pub(crate) struct RustcAllowIncoherentImplParser;
impl<S: Stage> NoArgsAttributeParser<S> for RustcAllowIncoherentImplParser {
    const PATH: &[Symbol] = &[sym::rustc_allow_incoherent_impl];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated_rustc_attr!(
        rustc_allow_incoherent_impl,
        "`#[rustc_allow_incoherent_impl]` has to be added to all impl items of an incoherent inherent impl"
    );

    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Method(MethodKind::Inherent))]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcAllowIncoherentImpl;
}

pub(crate) struct FundamentalParser;
impl<S: Stage> NoArgsAttributeParser<S> for FundamentalParser {
    const PATH: &[Symbol] = &[sym::fundamental];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const GATED: AttributeGate = gated!(fundamental);
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Struct), Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::Fundamental;
}
