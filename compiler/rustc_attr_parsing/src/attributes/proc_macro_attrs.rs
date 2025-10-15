use super::prelude::*;

const PROC_MACRO_ALLOWED_TARGETS: AllowedTargets =
    AllowedTargets::AllowList(&[Allow(Target::Fn), Warn(Target::Crate), Warn(Target::MacroCall)]);

pub(crate) struct ProcMacroParser;
impl<S: Stage> NoArgsAttributeParser<S> for ProcMacroParser {
    const PATH: &[Symbol] = &[sym::proc_macro];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = PROC_MACRO_ALLOWED_TARGETS;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ProcMacro;
}

pub(crate) struct ProcMacroAttributeParser;
impl<S: Stage> NoArgsAttributeParser<S> for ProcMacroAttributeParser {
    const PATH: &[Symbol] = &[sym::proc_macro_attribute];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = PROC_MACRO_ALLOWED_TARGETS;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::ProcMacroAttribute;
}

pub(crate) struct ProcMacroDeriveParser;
impl<S: Stage> SingleAttributeParser<S> for ProcMacroDeriveParser {
    const PATH: &[Symbol] = &[sym::proc_macro_derive];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = PROC_MACRO_ALLOWED_TARGETS;
    const TEMPLATE: AttributeTemplate = template!(
        List: &["TraitName", "TraitName, attributes(name1, name2, ...)"],
        "https://doc.rust-lang.org/reference/procedural-macros.html#derive-macros"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
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
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::MacroDef)]);
    const TEMPLATE: AttributeTemplate =
        template!(List: &["TraitName", "TraitName, attributes(name1, name2, ...)"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let (builtin_name, helper_attrs) = parse_derive_like(cx, args, false)?;
        Some(AttributeKind::RustcBuiltinMacro { builtin_name, helper_attrs, span: cx.attr_span })
    }
}

fn parse_derive_like<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    args: &ArgParser<'_>,
    trait_name_mandatory: bool,
) -> Option<(Option<Symbol>, ThinVec<Symbol>)> {
    let Some(list) = args.list() else {
        // For #[rustc_builtin_macro], it is permitted to leave out the trait name
        if args.no_args().is_ok() && !trait_name_mandatory {
            return Some((None, ThinVec::new()));
        }
        cx.expected_list(cx.attr_span);
        return None;
    };
    let mut items = list.mixed();

    // Parse the name of the trait that is derived.
    let Some(trait_attr) = items.next() else {
        cx.expected_at_least_one_argument(list.span);
        return None;
    };
    let Some(trait_attr) = trait_attr.meta_item() else {
        cx.unexpected_literal(trait_attr.span());
        return None;
    };
    let Some(trait_ident) = trait_attr.path().word() else {
        cx.expected_identifier(trait_attr.path().span());
        return None;
    };
    if !trait_ident.name.can_be_raw() {
        cx.expected_identifier(trait_ident.span);
        return None;
    }
    if let Err(e) = trait_attr.args().no_args() {
        cx.expected_no_args(e);
        return None;
    };

    // Parse optional attributes
    let mut attributes = ThinVec::new();
    if let Some(attrs) = items.next() {
        let Some(attr_list) = attrs.meta_item() else {
            cx.expected_list(attrs.span());
            return None;
        };
        if !attr_list.path().word_is(sym::attributes) {
            cx.expected_specific_argument(attrs.span(), &[sym::attributes]);
            return None;
        }
        let Some(attr_list) = attr_list.args().list() else {
            cx.expected_list(attrs.span());
            return None;
        };

        // Parse item in `attributes(...)` argument
        for attr in attr_list.mixed() {
            let Some(attr) = attr.meta_item() else {
                cx.expected_identifier(attr.span());
                return None;
            };
            if let Err(e) = attr.args().no_args() {
                cx.expected_no_args(e);
                return None;
            };
            let Some(ident) = attr.path().word() else {
                cx.expected_identifier(attr.path().span());
                return None;
            };
            if !ident.name.can_be_raw() {
                cx.expected_identifier(ident.span);
                return None;
            }
            attributes.push(ident.name);
        }
    }

    // If anything else is specified, we should reject it
    if let Some(next) = items.next() {
        cx.expected_no_args(next.span());
    }

    Some((Some(trait_ident.name), attributes))
}
