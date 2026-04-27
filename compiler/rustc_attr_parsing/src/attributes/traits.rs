use std::mem;

use super::prelude::*;
use crate::attributes::{NoArgsAttributeParser, SingleAttributeParser};
use crate::context::AcceptContext;
use crate::parser::ArgParser;
use crate::target_checking::AllowedTargets;
use crate::target_checking::Policy::{Allow, Warn};

pub(crate) struct RustcSkipDuringMethodDispatchParser;
impl SingleAttributeParser for RustcSkipDuringMethodDispatchParser {
    const PATH: &[Symbol] = &[sym::rustc_skip_during_method_dispatch];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);

    const TEMPLATE: AttributeTemplate = template!(List: &["array, boxed_slice"]);

    fn convert(cx: &mut AcceptContext<'_, '_>, args: &ArgParser) -> Option<AttributeKind> {
        let mut array = false;
        let mut boxed_slice = false;
        let args = cx.expect_list(args, cx.attr_span)?;
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
impl NoArgsAttributeParser for RustcParenSugarParser {
    const PATH: &[Symbol] = &[sym::rustc_paren_sugar];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcParenSugar;
}

// Markers

pub(crate) struct MarkerParser;
impl NoArgsAttributeParser for MarkerParser {
    const PATH: &[Symbol] = &[sym::marker];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Trait),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Marker;
}

pub(crate) struct RustcDenyExplicitImplParser;
impl NoArgsAttributeParser for RustcDenyExplicitImplParser {
    const PATH: &[Symbol] = &[sym::rustc_deny_explicit_impl];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcDenyExplicitImpl;
}

pub(crate) struct RustcDynIncompatibleTraitParser;
impl NoArgsAttributeParser for RustcDynIncompatibleTraitParser {
    const PATH: &[Symbol] = &[sym::rustc_dyn_incompatible_trait];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcDynIncompatibleTrait;
}

// Specialization

pub(crate) struct RustcSpecializationTraitParser;
impl NoArgsAttributeParser for RustcSpecializationTraitParser {
    const PATH: &[Symbol] = &[sym::rustc_specialization_trait];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcSpecializationTrait;
}

pub(crate) struct RustcUnsafeSpecializationMarkerParser;
impl NoArgsAttributeParser for RustcUnsafeSpecializationMarkerParser {
    const PATH: &[Symbol] = &[sym::rustc_unsafe_specialization_marker];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcUnsafeSpecializationMarker;
}

// Coherence

pub(crate) struct RustcCoinductiveParser;
impl NoArgsAttributeParser for RustcCoinductiveParser {
    const PATH: &[Symbol] = &[sym::rustc_coinductive];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcCoinductive;
}

pub(crate) struct RustcAllowIncoherentImplParser;
impl NoArgsAttributeParser for RustcAllowIncoherentImplParser {
    const PATH: &[Symbol] = &[sym::rustc_allow_incoherent_impl];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Method(MethodKind::Inherent))]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcAllowIncoherentImpl;
}

pub(crate) struct FundamentalParser;
impl NoArgsAttributeParser for FundamentalParser {
    const PATH: &[Symbol] = &[sym::fundamental];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Struct), Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::Fundamental;
}
