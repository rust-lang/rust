use rustc_errors::Diagnostic;
use rustc_session::lint::builtin::AMBIGUOUS_DERIVE_HELPERS;

use super::prelude::*;

const PROC_MACRO_ALLOWED_TARGETS: AllowedTargets =
    AllowedTargets::AllowList(&[Allow(Target::Fn), Warn(Target::Crate), Warn(Target::MacroCall)]);

pub(crate) struct ProcMacroParser;
impl<S: Stage> NoArgsAttributeParser<S> for ProcMacroParser {
    const PATH: &[Symbol] = &[sym::proc_macro];
    const ALLOWED_TARGETS: AllowedTargets = PROC_MACRO_ALLOWED_TARGETS;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ProcMacro;
}

pub(crate) struct ProcMacroAttributeParser;
impl<S: Stage> NoArgsAttributeParser<S> for ProcMacroAttributeParser {
    const PATH: &[Symbol] = &[sym::proc_macro_attribute];
    const ALLOWED_TARGETS: AllowedTargets = PROC_MACRO_ALLOWED_TARGETS;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ProcMacroAttribute;
}

pub(crate) struct ProcMacroDeriveParser;
impl<S: Stage> SingleAttributeParser<S> for ProcMacroDeriveParser {
    const PATH: &[Symbol] = &[sym::proc_macro_derive];
    const ALLOWED_TARGETS: AllowedTargets = PROC_MACRO_ALLOWED_TARGETS;
    const TEMPLATE: AttributeTemplate = template!(
        List: &["TraitName", "TraitName, attributes(name1, name2, ...)"],
        "https://doc.rust-lang.org/reference/procedural-macros.html#derive-macros"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let (trait_name, helper_attrs) = parse_derive_like(cx, args, true)?;
        Some(AttributeKind::ProcMacroDerive {
            trait_name: trait_name.expect("Trait name is mandatory, so it is present"),
            helper_attrs,
            span: cx.attr_span,
        })
    }
}

pub(crate) struct RustcBuiltinMacroParser;
impl<S: Stage> SingleAttributeParser<S> for RustcBuiltinMacroParser {
    const PATH: &[Symbol] = &[sym::rustc_builtin_macro];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::MacroDef)]);
    const TEMPLATE: AttributeTemplate =
        template!(List: &["TraitName", "TraitName, attributes(name1, name2, ...)"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let (builtin_name, helper_attrs) = parse_derive_like(cx, args, false)?;
        Some(AttributeKind::RustcBuiltinMacro { builtin_name, helper_attrs, span: cx.attr_span })
    }
}

fn parse_derive_like<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    args: &ArgParser,
    trait_name_mandatory: bool,
) -> Option<(Option<Symbol>, ThinVec<Symbol>)> {
    let Some(list) = args.as_list() else {
        // For #[rustc_builtin_macro], it is permitted to leave out the trait name
        if args.no_args().is_ok() && !trait_name_mandatory {
            return Some((None, ThinVec::new()));
        }
        let attr_span = cx.attr_span;
        cx.adcx().expected_list(attr_span, args);
        return None;
    };
    let mut items = list.mixed();

    // Parse the name of the trait that is derived.
    let Some(trait_attr) = items.next() else {
        cx.adcx().expected_at_least_one_argument(list.span);
        return None;
    };
    let Some(trait_attr) = trait_attr.as_meta_item() else {
        cx.adcx().expected_not_literal(trait_attr.span());
        return None;
    };
    let Some(trait_ident) = trait_attr.path().word() else {
        cx.adcx().expected_identifier(trait_attr.path().span());
        return None;
    };
    if !trait_ident.name.can_be_raw() {
        cx.adcx().expected_identifier(trait_ident.span);
        return None;
    }
    if let Err(e) = trait_attr.args().no_args() {
        cx.adcx().expected_no_args(e);
        return None;
    };

    // Parse optional attributes
    let mut attributes = ThinVec::new();
    if let Some(attrs) = items.next() {
        let Some(attr_list) = attrs.as_meta_item() else {
            cx.adcx().expected_not_literal(attrs.span());
            return None;
        };
        if !attr_list.path().word_is(sym::attributes) {
            cx.adcx().expected_specific_argument(attrs.span(), &[sym::attributes]);
            return None;
        }
        let attr_list = cx.expect_list(attr_list.args(), attrs.span())?;

        // Parse item in `attributes(...)` argument
        for attr in attr_list.mixed() {
            let Some(attr) = attr.as_meta_item() else {
                cx.adcx().expected_identifier(attr.span());
                return None;
            };
            if let Err(e) = attr.args().no_args() {
                cx.adcx().expected_no_args(e);
                return None;
            };
            let Some(ident) = attr.path().word() else {
                cx.adcx().expected_identifier(attr.path().span());
                return None;
            };
            if !ident.name.can_be_raw() {
                cx.adcx().expected_identifier(ident.span);
                return None;
            }
            if rustc_feature::is_builtin_attr_name(ident.name) {
                cx.emit_dyn_lint(
                    AMBIGUOUS_DERIVE_HELPERS,
                    |dcx, level| crate::errors::AmbiguousDeriveHelpers.into_diag(dcx, level),
                    ident.span,
                );
            }
            attributes.push(ident.name);
        }
    }

    // If anything else is specified, we should reject it
    if let Some(next) = items.next() {
        cx.adcx().expected_no_args(next.span());
    }

    Some((Some(trait_ident.name), attributes))
}
