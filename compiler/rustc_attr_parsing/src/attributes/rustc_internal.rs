use rustc_ast::{LitIntType, LitKind, MetaItemLit};
use rustc_session::errors;

use super::prelude::*;
use super::util::parse_single_integer;
use crate::session_diagnostics::RustcScalableVectorCountOutOfRange;

pub(crate) struct RustcMainParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcMainParser {
    const PATH: &'static [Symbol] = &[sym::rustc_main];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcMain;
}

pub(crate) struct RustcMustImplementOneOfParser;

impl<S: Stage> SingleAttributeParser<S> for RustcMustImplementOneOfParser {
    const PATH: &[Symbol] = &[sym::rustc_must_implement_one_of];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const TEMPLATE: AttributeTemplate = template!(List: &["function1, function2, ..."]);
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };

        let mut fn_names = ThinVec::new();

        let inputs: Vec<_> = list.mixed().collect();

        if inputs.len() < 2 {
            cx.expected_list_with_num_args_or_more(2, list.span);
            return None;
        }

        let mut errored = false;
        for argument in inputs {
            let Some(meta) = argument.meta_item() else {
                cx.expected_identifier(argument.span());
                return None;
            };

            let Some(ident) = meta.ident() else {
                cx.dcx().emit_err(errors::MustBeNameOfAssociatedFunction { span: meta.span() });
                errored = true;
                continue;
            };

            fn_names.push(ident);
        }
        if errored {
            return None;
        }

        Some(AttributeKind::RustcMustImplementOneOf { attr_span: cx.attr_span, fn_names })
    }
}

pub(crate) struct RustcNeverReturnsNullPointerParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcNeverReturnsNullPointerParser {
    const PATH: &[Symbol] = &[sym::rustc_never_returns_null_ptr];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcNeverReturnsNullPointer;
}
pub(crate) struct RustcNoImplicitAutorefsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcNoImplicitAutorefsParser {
    const PATH: &[Symbol] = &[sym::rustc_no_implicit_autorefs];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);

    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcNoImplicitAutorefs;
}

pub(crate) struct RustcLayoutScalarValidRangeStartParser;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeStartParser {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_start];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["start"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        parse_single_integer(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeStart(Box::new(n), cx.attr_span))
    }
}

pub(crate) struct RustcLayoutScalarValidRangeEndParser;

impl<S: Stage> SingleAttributeParser<S> for RustcLayoutScalarValidRangeEndParser {
    const PATH: &'static [Symbol] = &[sym::rustc_layout_scalar_valid_range_end];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["end"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        parse_single_integer(cx, args)
            .map(|n| AttributeKind::RustcLayoutScalarValidRangeEnd(Box::new(n), cx.attr_span))
    }
}

pub(crate) struct RustcLegacyConstGenericsParser;

impl<S: Stage> SingleAttributeParser<S> for RustcLegacyConstGenericsParser {
    const PATH: &[Symbol] = &[sym::rustc_legacy_const_generics];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["N"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::List(meta_items) = args else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };

        let mut parsed_indexes = ThinVec::new();
        let mut errored = false;

        for possible_index in meta_items.mixed() {
            if let MetaItemOrLitParser::Lit(MetaItemLit {
                kind: LitKind::Int(index, LitIntType::Unsuffixed),
                ..
            }) = possible_index
            {
                parsed_indexes.push((index.0 as usize, possible_index.span()));
            } else {
                cx.expected_integer_literal(possible_index.span());
                errored = true;
            }
        }
        if errored {
            return None;
        } else if parsed_indexes.is_empty() {
            cx.expected_at_least_one_argument(args.span()?);
            return None;
        }

        Some(AttributeKind::RustcLegacyConstGenerics {
            fn_indexes: parsed_indexes,
            attr_span: cx.attr_span,
        })
    }
}

pub(crate) struct RustcLintDiagnosticsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcLintDiagnosticsParser {
    const PATH: &[Symbol] = &[sym::rustc_lint_diagnostics];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcLintDiagnostics;
}

pub(crate) struct RustcLintOptDenyFieldAccessParser;

impl<S: Stage> SingleAttributeParser<S> for RustcLintOptDenyFieldAccessParser {
    const PATH: &[Symbol] = &[sym::rustc_lint_opt_deny_field_access];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Field)]);
    const TEMPLATE: AttributeTemplate = template!(Word);
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(arg) = args.list().and_then(MetaItemListParser::single) else {
            cx.expected_single_argument(cx.attr_span);
            return None;
        };

        let MetaItemOrLitParser::Lit(MetaItemLit { kind: LitKind::Str(lint_message, _), .. }) = arg
        else {
            cx.expected_string_literal(arg.span(), arg.lit());
            return None;
        };

        Some(AttributeKind::RustcLintOptDenyFieldAccess { lint_message: *lint_message })
    }
}

pub(crate) struct RustcLintOptTyParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcLintOptTyParser {
    const PATH: &[Symbol] = &[sym::rustc_lint_opt_ty];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcLintOptTy;
}

pub(crate) struct RustcLintQueryInstabilityParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcLintQueryInstabilityParser {
    const PATH: &[Symbol] = &[sym::rustc_lint_query_instability];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcLintQueryInstability;
}

pub(crate) struct RustcLintUntrackedQueryInformationParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcLintUntrackedQueryInformationParser {
    const PATH: &[Symbol] = &[sym::rustc_lint_untracked_query_information];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);

    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcLintUntrackedQueryInformation;
}

pub(crate) struct RustcObjectLifetimeDefaultParser;

impl<S: Stage> SingleAttributeParser<S> for RustcObjectLifetimeDefaultParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_object_lifetime_default];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(Word);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }

        Some(AttributeKind::RustcObjectLifetimeDefault)
    }
}

pub(crate) struct RustcSimdMonomorphizeLaneLimitParser;

impl<S: Stage> SingleAttributeParser<S> for RustcSimdMonomorphizeLaneLimitParser {
    const PATH: &[Symbol] = &[sym::rustc_simd_monomorphize_lane_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        Some(AttributeKind::RustcSimdMonomorphizeLaneLimit(cx.parse_limit_int(nv)?))
    }
}

pub(crate) struct RustcScalableVectorParser;

impl<S: Stage> SingleAttributeParser<S> for RustcScalableVectorParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_scalable_vector];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const TEMPLATE: AttributeTemplate = template!(Word, List: &["count"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if args.no_args().is_ok() {
            return Some(AttributeKind::RustcScalableVector {
                element_count: None,
                span: cx.attr_span,
            });
        }

        let n = parse_single_integer(cx, args)?;
        let Ok(n) = n.try_into() else {
            cx.emit_err(RustcScalableVectorCountOutOfRange { span: cx.attr_span, n });
            return None;
        };
        Some(AttributeKind::RustcScalableVector { element_count: Some(n), span: cx.attr_span })
    }
}
