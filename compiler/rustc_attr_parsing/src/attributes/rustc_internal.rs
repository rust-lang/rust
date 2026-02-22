use std::path::PathBuf;

use rustc_ast::{LitIntType, LitKind, MetaItemLit};
use rustc_hir::LangItem;
use rustc_hir::attrs::{
    BorrowckGraphvizFormatKind, CguFields, CguKind, DivergingBlockBehavior,
    DivergingFallbackBehavior, RustcCleanAttribute, RustcCleanQueries, RustcLayoutType,
    RustcMirKind,
};
use rustc_session::errors;
use rustc_span::Symbol;

use super::prelude::*;
use super::util::parse_single_integer;
use crate::session_diagnostics::{
    AttributeRequiresOpt, CguFieldsMissing, RustcScalableVectorCountOutOfRange, UnknownLangItem,
};

pub(crate) struct RustcMainParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcMainParser {
    const PATH: &[Symbol] = &[sym::rustc_main];
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
    const PATH: &[Symbol] = &[sym::rustc_layout_scalar_valid_range_start];
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
    const PATH: &[Symbol] = &[sym::rustc_layout_scalar_valid_range_end];
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

fn parse_cgu_fields<S: Stage>(
    cx: &mut AcceptContext<'_, '_, S>,
    args: &ArgParser,
    accepts_kind: bool,
) -> Option<(Symbol, Symbol, Option<CguKind>)> {
    let Some(args) = args.list() else {
        cx.expected_list(cx.attr_span, args);
        return None;
    };

    let mut cfg = None::<(Symbol, Span)>;
    let mut module = None::<(Symbol, Span)>;
    let mut kind = None::<(Symbol, Span)>;

    for arg in args.mixed() {
        let Some(arg) = arg.meta_item() else {
            cx.expected_name_value(args.span, None);
            continue;
        };

        let res = match arg.ident().map(|i| i.name) {
            Some(sym::cfg) => &mut cfg,
            Some(sym::module) => &mut module,
            Some(sym::kind) if accepts_kind => &mut kind,
            _ => {
                cx.expected_specific_argument(
                    arg.path().span(),
                    if accepts_kind {
                        &[sym::cfg, sym::module, sym::kind]
                    } else {
                        &[sym::cfg, sym::module]
                    },
                );
                continue;
            }
        };

        let Some(i) = arg.args().name_value() else {
            cx.expected_name_value(arg.span(), None);
            continue;
        };

        let Some(str) = i.value_as_str() else {
            cx.expected_string_literal(i.value_span, Some(i.value_as_lit()));
            continue;
        };

        if res.is_some() {
            cx.duplicate_key(arg.span(), arg.ident().unwrap().name);
            continue;
        }

        *res = Some((str, i.value_span));
    }

    let Some((cfg, _)) = cfg else {
        cx.emit_err(CguFieldsMissing { span: args.span, name: &cx.attr_path, field: sym::cfg });
        return None;
    };
    let Some((module, _)) = module else {
        cx.emit_err(CguFieldsMissing { span: args.span, name: &cx.attr_path, field: sym::module });
        return None;
    };
    let kind = if let Some((kind, span)) = kind {
        Some(match kind {
            sym::no => CguKind::No,
            sym::pre_dash_lto => CguKind::PreDashLto,
            sym::post_dash_lto => CguKind::PostDashLto,
            sym::any => CguKind::Any,
            _ => {
                cx.expected_specific_argument_strings(
                    span,
                    &[sym::no, sym::pre_dash_lto, sym::post_dash_lto, sym::any],
                );
                return None;
            }
        })
    } else {
        // return None so that an unwrap for the attributes that need it is ok.
        if accepts_kind {
            cx.emit_err(CguFieldsMissing {
                span: args.span,
                name: &cx.attr_path,
                field: sym::kind,
            });
            return None;
        };

        None
    };

    Some((cfg, module, kind))
}

#[derive(Default)]
pub(crate) struct RustcCguTestAttributeParser {
    items: ThinVec<(Span, CguFields)>,
}

impl<S: Stage> AttributeParser<S> for RustcCguTestAttributeParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[
        (
            &[sym::rustc_partition_reused],
            template!(List: &[r#"cfg = "...", module = "...""#]),
            |this, cx, args| {
                this.items.extend(parse_cgu_fields(cx, args, false).map(|(cfg, module, _)| {
                    (cx.attr_span, CguFields::PartitionReused { cfg, module })
                }));
            },
        ),
        (
            &[sym::rustc_partition_codegened],
            template!(List: &[r#"cfg = "...", module = "...""#]),
            |this, cx, args| {
                this.items.extend(parse_cgu_fields(cx, args, false).map(|(cfg, module, _)| {
                    (cx.attr_span, CguFields::PartitionCodegened { cfg, module })
                }));
            },
        ),
        (
            &[sym::rustc_expected_cgu_reuse],
            template!(List: &[r#"cfg = "...", module = "...", kind = "...""#]),
            |this, cx, args| {
                this.items.extend(parse_cgu_fields(cx, args, true).map(|(cfg, module, kind)| {
                    // unwrap ok because if not given, we return None in `parse_cgu_fields`.
                    (cx.attr_span, CguFields::ExpectedCguReuse { cfg, module, kind: kind.unwrap() })
                }));
            },
        ),
    ];

    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Mod), Allow(Target::Crate)]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        Some(AttributeKind::RustcCguTestAttr(self.items))
    }
}

pub(crate) struct RustcDeprecatedSafe2024Parser;

impl<S: Stage> SingleAttributeParser<S> for RustcDeprecatedSafe2024Parser {
    const PATH: &[Symbol] = &[sym::rustc_deprecated_safe_2024];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const TEMPLATE: AttributeTemplate = template!(List: &[r#"audit_that = "...""#]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(args) = args.list() else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };

        let Some(single) = args.single() else {
            cx.expected_single_argument(args.span);
            return None;
        };

        let Some(arg) = single.meta_item() else {
            cx.expected_name_value(args.span, None);
            return None;
        };

        let Some(args) = arg.word_is(sym::audit_that) else {
            cx.expected_specific_argument(arg.span(), &[sym::audit_that]);
            return None;
        };

        let Some(nv) = args.name_value() else {
            cx.expected_name_value(arg.span(), Some(sym::audit_that));
            return None;
        };

        let Some(suggestion) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        Some(AttributeKind::RustcDeprecatedSafe2024 { suggestion })
    }
}

pub(crate) struct RustcConversionSuggestionParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcConversionSuggestionParser {
    const PATH: &[Symbol] = &[sym::rustc_conversion_suggestion];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcConversionSuggestion;
}

pub(crate) struct RustcCaptureAnalysisParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcCaptureAnalysisParser {
    const PATH: &[Symbol] = &[sym::rustc_capture_analysis];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Closure)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcCaptureAnalysis;
}

pub(crate) struct RustcNeverTypeOptionsParser;

impl<S: Stage> SingleAttributeParser<S> for RustcNeverTypeOptionsParser {
    const PATH: &[Symbol] = &[sym::rustc_never_type_options];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const TEMPLATE: AttributeTemplate = template!(List: &[
        r#"fallback = "unit", "never", "no""#,
        r#"diverging_block_default = "unit", "never""#,
    ]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };

        let mut fallback = None::<Ident>;
        let mut diverging_block_default = None::<Ident>;

        for arg in list.mixed() {
            let Some(meta) = arg.meta_item() else {
                cx.expected_name_value(arg.span(), None);
                continue;
            };

            let res = match meta.ident().map(|i| i.name) {
                Some(sym::fallback) => &mut fallback,
                Some(sym::diverging_block_default) => &mut diverging_block_default,
                _ => {
                    cx.expected_specific_argument(
                        meta.path().span(),
                        &[sym::fallback, sym::diverging_block_default],
                    );
                    continue;
                }
            };

            let Some(nv) = meta.args().name_value() else {
                cx.expected_name_value(meta.span(), None);
                continue;
            };

            let Some(field) = nv.value_as_str() else {
                cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
                continue;
            };

            if res.is_some() {
                cx.duplicate_key(meta.span(), meta.ident().unwrap().name);
                continue;
            }

            *res = Some(Ident { name: field, span: nv.value_span });
        }

        let fallback = match fallback {
            None => None,
            Some(Ident { name: sym::unit, .. }) => Some(DivergingFallbackBehavior::ToUnit),
            Some(Ident { name: sym::never, .. }) => Some(DivergingFallbackBehavior::ToNever),
            Some(Ident { name: sym::no, .. }) => Some(DivergingFallbackBehavior::NoFallback),
            Some(Ident { span, .. }) => {
                cx.expected_specific_argument_strings(span, &[sym::unit, sym::never, sym::no]);
                return None;
            }
        };

        let diverging_block_default = match diverging_block_default {
            None => None,
            Some(Ident { name: sym::unit, .. }) => Some(DivergingBlockBehavior::Unit),
            Some(Ident { name: sym::never, .. }) => Some(DivergingBlockBehavior::Never),
            Some(Ident { span, .. }) => {
                cx.expected_specific_argument_strings(span, &[sym::unit, sym::no]);
                return None;
            }
        };

        Some(AttributeKind::RustcNeverTypeOptions { fallback, diverging_block_default })
    }
}

pub(crate) struct RustcTrivialFieldReadsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcTrivialFieldReadsParser {
    const PATH: &[Symbol] = &[sym::rustc_trivial_field_reads];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Trait)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcTrivialFieldReads;
}

pub(crate) struct RustcNoMirInlineParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcNoMirInlineParser {
    const PATH: &[Symbol] = &[sym::rustc_no_mir_inline];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcNoMirInline;
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

pub(crate) struct RustcRegionsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcRegionsParser {
    const PATH: &[Symbol] = &[sym::rustc_regions];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);

    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcRegions;
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

impl<S: Stage> NoArgsAttributeParser<S> for RustcObjectLifetimeDefaultParser {
    const PATH: &[Symbol] = &[sym::rustc_object_lifetime_default];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcObjectLifetimeDefault;
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
    const PATH: &[Symbol] = &[sym::rustc_scalable_vector];
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

pub(crate) struct LangParser;

impl<S: Stage> SingleAttributeParser<S> for LangParser {
    const PATH: &[Symbol] = &[sym::lang];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS); // Targets are checked per lang item in `rustc_passes`
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "name");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        let Some(name) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };
        let Some(lang_item) = LangItem::from_name(name) else {
            cx.emit_err(UnknownLangItem { span: cx.attr_span, name });
            return None;
        };
        Some(AttributeKind::Lang(lang_item, cx.attr_span))
    }
}

pub(crate) struct RustcHasIncoherentInherentImplsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcHasIncoherentInherentImplsParser {
    const PATH: &[Symbol] = &[sym::rustc_has_incoherent_inherent_impls];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Trait),
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::ForeignTy),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcHasIncoherentInherentImpls;
}

pub(crate) struct PanicHandlerParser;

impl<S: Stage> NoArgsAttributeParser<S> for PanicHandlerParser {
    const PATH: &[Symbol] = &[sym::panic_handler];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS); // Targets are checked per lang item in `rustc_passes`
    const CREATE: fn(Span) -> AttributeKind = |span| AttributeKind::Lang(LangItem::PanicImpl, span);
}

pub(crate) struct RustcHiddenTypeOfOpaquesParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcHiddenTypeOfOpaquesParser {
    const PATH: &[Symbol] = &[sym::rustc_hidden_type_of_opaques];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcHiddenTypeOfOpaques;
}
pub(crate) struct RustcNounwindParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcNounwindParser {
    const PATH: &[Symbol] = &[sym::rustc_nounwind];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::ForeignFn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcNounwind;
}

pub(crate) struct RustcOffloadKernelParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcOffloadKernelParser {
    const PATH: &[Symbol] = &[sym::rustc_offload_kernel];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcOffloadKernel;
}

pub(crate) struct RustcLayoutParser;

impl<S: Stage> CombineAttributeParser<S> for RustcLayoutParser {
    const PATH: &[Symbol] = &[sym::rustc_layout];

    type Item = RustcLayoutType;

    const CONVERT: ConvertFn<Self::Item> = |items, _| AttributeKind::RustcLayout(items);

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::TyAlias),
    ]);

    const TEMPLATE: AttributeTemplate =
        template!(List: &["abi", "align", "size", "homogenous_aggregate", "debug"]);
    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let ArgParser::List(items) = args else {
            cx.expected_list(cx.attr_span, args);
            return vec![];
        };

        let mut result = Vec::new();
        for item in items.mixed() {
            let Some(arg) = item.meta_item() else {
                cx.unexpected_literal(item.span());
                continue;
            };
            let Some(ident) = arg.ident() else {
                cx.expected_identifier(arg.span());
                return vec![];
            };
            let ty = match ident.name {
                sym::abi => RustcLayoutType::Abi,
                sym::align => RustcLayoutType::Align,
                sym::size => RustcLayoutType::Size,
                sym::homogeneous_aggregate => RustcLayoutType::HomogenousAggregate,
                sym::debug => RustcLayoutType::Debug,
                _ => {
                    cx.expected_specific_argument(
                        ident.span,
                        &[sym::abi, sym::align, sym::size, sym::homogeneous_aggregate, sym::debug],
                    );
                    continue;
                }
            };
            result.push(ty);
        }
        result
    }
}

pub(crate) struct RustcMirParser;

impl<S: Stage> CombineAttributeParser<S> for RustcMirParser {
    const PATH: &[Symbol] = &[sym::rustc_mir];

    type Item = RustcMirKind;

    const CONVERT: ConvertFn<Self::Item> = |items, _| AttributeKind::RustcMir(items);

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
    ]);

    const TEMPLATE: AttributeTemplate = template!(List: &["arg1, arg2, ..."]);

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span, args);
            return ThinVec::new();
        };

        list.mixed()
            .filter_map(|arg| arg.meta_item())
            .filter_map(|mi| {
                if let Some(ident) = mi.ident() {
                    match ident.name {
                        sym::rustc_peek_maybe_init => Some(RustcMirKind::PeekMaybeInit),
                        sym::rustc_peek_maybe_uninit => Some(RustcMirKind::PeekMaybeUninit),
                        sym::rustc_peek_liveness => Some(RustcMirKind::PeekLiveness),
                        sym::stop_after_dataflow => Some(RustcMirKind::StopAfterDataflow),
                        sym::borrowck_graphviz_postflow => {
                            let Some(nv) = mi.args().name_value() else {
                                cx.expected_name_value(
                                    mi.span(),
                                    Some(sym::borrowck_graphviz_postflow),
                                );
                                return None;
                            };
                            let Some(path) = nv.value_as_str() else {
                                cx.expected_string_literal(nv.value_span, None);
                                return None;
                            };
                            let path = PathBuf::from(path.to_string());
                            if path.file_name().is_some() {
                                Some(RustcMirKind::BorrowckGraphvizPostflow { path })
                            } else {
                                cx.expected_filename_literal(nv.value_span);
                                None
                            }
                        }
                        sym::borrowck_graphviz_format => {
                            let Some(nv) = mi.args().name_value() else {
                                cx.expected_name_value(
                                    mi.span(),
                                    Some(sym::borrowck_graphviz_format),
                                );
                                return None;
                            };
                            let Some(format) = nv.value_as_ident() else {
                                cx.expected_identifier(nv.value_span);
                                return None;
                            };
                            match format.name {
                                sym::two_phase => Some(RustcMirKind::BorrowckGraphvizFormat {
                                    format: BorrowckGraphvizFormatKind::TwoPhase,
                                }),
                                _ => {
                                    cx.expected_specific_argument(format.span, &[sym::two_phase]);
                                    None
                                }
                            }
                        }
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .collect()
    }
}
pub(crate) struct RustcNonConstTraitMethodParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcNonConstTraitMethodParser {
    const PATH: &[Symbol] = &[sym::rustc_non_const_trait_method];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::Trait { body: false })),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcNonConstTraitMethod;
}

pub(crate) struct RustcCleanParser;

impl<S: Stage> CombineAttributeParser<S> for RustcCleanParser {
    const PATH: &[Symbol] = &[sym::rustc_clean];

    type Item = RustcCleanAttribute;

    const CONVERT: ConvertFn<Self::Item> = |items, _| AttributeKind::RustcClean(items);

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        // tidy-alphabetical-start
        Allow(Target::AssocConst),
        Allow(Target::AssocTy),
        Allow(Target::Const),
        Allow(Target::Enum),
        Allow(Target::Expression),
        Allow(Target::Field),
        Allow(Target::Fn),
        Allow(Target::ForeignMod),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Mod),
        Allow(Target::Static),
        Allow(Target::Struct),
        Allow(Target::Trait),
        Allow(Target::TyAlias),
        Allow(Target::Union),
        // tidy-alphabetical-end
    ]);

    const TEMPLATE: AttributeTemplate =
        template!(List: &[r#"cfg = "...", /*opt*/ label = "...", /*opt*/ except = "...""#]);

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        if !cx.cx.sess.opts.unstable_opts.query_dep_graph {
            cx.emit_err(AttributeRequiresOpt { span: cx.attr_span, opt: "-Z query-dep-graph" });
        }
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span, args);
            return None;
        };
        let mut except = None;
        let mut loaded_from_disk = None;
        let mut cfg = None;

        for item in list.mixed() {
            let Some((value, name)) =
                item.meta_item().and_then(|m| Option::zip(m.args().name_value(), m.ident()))
            else {
                cx.expected_name_value(item.span(), None);
                continue;
            };
            let value_span = value.value_span;
            let Some(value) = value.value_as_str() else {
                cx.expected_string_literal(value_span, None);
                continue;
            };
            match name.name {
                sym::cfg if cfg.is_some() => {
                    cx.duplicate_key(item.span(), sym::cfg);
                }

                sym::cfg => {
                    cfg = Some(value);
                }
                sym::except if except.is_some() => {
                    cx.duplicate_key(item.span(), sym::except);
                }
                sym::except => {
                    let entries =
                        value.as_str().split(',').map(|s| Symbol::intern(s.trim())).collect();
                    except = Some(RustcCleanQueries { entries, span: value_span });
                }
                sym::loaded_from_disk if loaded_from_disk.is_some() => {
                    cx.duplicate_key(item.span(), sym::loaded_from_disk);
                }
                sym::loaded_from_disk => {
                    let entries =
                        value.as_str().split(',').map(|s| Symbol::intern(s.trim())).collect();
                    loaded_from_disk = Some(RustcCleanQueries { entries, span: value_span });
                }
                _ => {
                    cx.expected_specific_argument(
                        name.span,
                        &[sym::cfg, sym::except, sym::loaded_from_disk],
                    );
                }
            }
        }
        let Some(cfg) = cfg else {
            cx.expected_specific_argument(list.span, &[sym::cfg]);
            return None;
        };

        Some(RustcCleanAttribute { span: cx.attr_span, cfg, except, loaded_from_disk })
    }
}

pub(crate) struct RustcIfThisChangedParser;

impl<S: Stage> SingleAttributeParser<S> for RustcIfThisChangedParser {
    const PATH: &[Symbol] = &[sym::rustc_if_this_changed];

    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;

    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        // tidy-alphabetical-start
        Allow(Target::AssocConst),
        Allow(Target::AssocTy),
        Allow(Target::Const),
        Allow(Target::Enum),
        Allow(Target::Expression),
        Allow(Target::Field),
        Allow(Target::Fn),
        Allow(Target::ForeignMod),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Mod),
        Allow(Target::Static),
        Allow(Target::Struct),
        Allow(Target::Trait),
        Allow(Target::TyAlias),
        Allow(Target::Union),
        // tidy-alphabetical-end
    ]);

    const TEMPLATE: AttributeTemplate = template!(Word, List: &["DepNode"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if !cx.cx.sess.opts.unstable_opts.query_dep_graph {
            cx.emit_err(AttributeRequiresOpt { span: cx.attr_span, opt: "-Z query-dep-graph" });
        }
        match args {
            ArgParser::NoArgs => Some(AttributeKind::RustcIfThisChanged(cx.attr_span, None)),
            ArgParser::List(list) => {
                let Some(item) = list.single() else {
                    cx.expected_single_argument(list.span);
                    return None;
                };
                let Some(ident) = item.meta_item().and_then(|item| item.ident()) else {
                    cx.expected_identifier(item.span());
                    return None;
                };
                Some(AttributeKind::RustcIfThisChanged(cx.attr_span, Some(ident.name)))
            }
            ArgParser::NameValue(_) => {
                cx.expected_list_or_no_args(cx.inner_span);
                None
            }
        }
    }
}

pub(crate) struct RustcThenThisWouldNeedParser;

impl<S: Stage> CombineAttributeParser<S> for RustcThenThisWouldNeedParser {
    const PATH: &[Symbol] = &[sym::rustc_then_this_would_need];
    type Item = Ident;

    const CONVERT: ConvertFn<Self::Item> =
        |items, span| AttributeKind::RustcThenThisWouldNeed(span, items);
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        // tidy-alphabetical-start
        Allow(Target::AssocConst),
        Allow(Target::AssocTy),
        Allow(Target::Const),
        Allow(Target::Enum),
        Allow(Target::Expression),
        Allow(Target::Field),
        Allow(Target::Fn),
        Allow(Target::ForeignMod),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Mod),
        Allow(Target::Static),
        Allow(Target::Struct),
        Allow(Target::Trait),
        Allow(Target::TyAlias),
        Allow(Target::Union),
        // tidy-alphabetical-end
    ]);

    const TEMPLATE: AttributeTemplate = template!(List: &["DepNode"]);

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        if !cx.cx.sess.opts.unstable_opts.query_dep_graph {
            cx.emit_err(AttributeRequiresOpt { span: cx.attr_span, opt: "-Z query-dep-graph" });
        }
        let Some(item) = args.list().and_then(|l| l.single()) else {
            cx.expected_single_argument(cx.inner_span);
            return None;
        };
        let Some(ident) = item.meta_item().and_then(|item| item.ident()) else {
            cx.expected_identifier(item.span());
            return None;
        };
        Some(ident)
    }
}

pub(crate) struct RustcInsignificantDtorParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcInsignificantDtorParser {
    const PATH: &[Symbol] = &[sym::rustc_insignificant_dtor];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Enum),
        Allow(Target::Struct),
        Allow(Target::ForeignTy),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcInsignificantDtor;
}

pub(crate) struct RustcEffectiveVisibilityParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcEffectiveVisibilityParser {
    const PATH: &[Symbol] = &[sym::rustc_effective_visibility];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Use),
        Allow(Target::Static),
        Allow(Target::Const),
        Allow(Target::Fn),
        Allow(Target::Closure),
        Allow(Target::Mod),
        Allow(Target::ForeignMod),
        Allow(Target::TyAlias),
        Allow(Target::Enum),
        Allow(Target::Variant),
        Allow(Target::Struct),
        Allow(Target::Field),
        Allow(Target::Union),
        Allow(Target::Trait),
        Allow(Target::TraitAlias),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::AssocConst),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::AssocTy),
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Allow(Target::ForeignTy),
        Allow(Target::MacroDef),
        Allow(Target::PatField),
        Allow(Target::Crate),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcEffectiveVisibility;
}

pub(crate) struct RustcDiagnosticItemParser;

impl<S: Stage> SingleAttributeParser<S> for RustcDiagnosticItemParser {
    const PATH: &[Symbol] = &[sym::rustc_diagnostic_item];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Trait),
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::MacroDef),
        Allow(Target::TyAlias),
        Allow(Target::AssocTy),
        Allow(Target::AssocConst),
        Allow(Target::Fn),
        Allow(Target::Const),
        Allow(Target::Mod),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Crate),
    ]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "name");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        let Some(value) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };
        Some(AttributeKind::RustcDiagnosticItem(value))
    }
}

pub(crate) struct RustcDoNotConstCheckParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDoNotConstCheckParser {
    const PATH: &[Symbol] = &[sym::rustc_do_not_const_check];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
    ]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDoNotConstCheck;
}

pub(crate) struct RustcNonnullOptimizationGuaranteedParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcNonnullOptimizationGuaranteedParser {
    const PATH: &[Symbol] = &[sym::rustc_nonnull_optimization_guaranteed];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Struct)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcNonnullOptimizationGuaranteed;
}

pub(crate) struct RustcSymbolName;

impl<S: Stage> SingleAttributeParser<S> for RustcSymbolName {
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Allow(Target::Impl { of_trait: false }),
    ]);
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const PATH: &[Symbol] = &[sym::rustc_symbol_name];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const TEMPLATE: AttributeTemplate = template!(Word);
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }
        Some(AttributeKind::RustcSymbolName(cx.attr_span))
    }
}

pub(crate) struct RustcDefPath;

impl<S: Stage> SingleAttributeParser<S> for RustcDefPath {
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Allow(Target::Impl { of_trait: false }),
    ]);
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const PATH: &[Symbol] = &[sym::rustc_def_path];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const TEMPLATE: AttributeTemplate = template!(Word);
    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        if let Err(span) = args.no_args() {
            cx.expected_no_args(span);
            return None;
        }
        Some(AttributeKind::RustcDefPath(cx.attr_span))
    }
}

pub(crate) struct RustcStrictCoherenceParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcStrictCoherenceParser {
    const PATH: &[Symbol] = &[sym::rustc_strict_coherence];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Trait),
        Allow(Target::Struct),
        Allow(Target::Enum),
        Allow(Target::Union),
        Allow(Target::ForeignTy),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcStrictCoherence;
}

pub(crate) struct RustcReservationImplParser;

impl<S: Stage> SingleAttributeParser<S> for RustcReservationImplParser {
    const PATH: &[Symbol] = &[sym::rustc_reservation_impl];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Impl { of_trait: true })]);

    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "reservation message");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(args.span().unwrap_or(cx.attr_span), None);
            return None;
        };

        let Some(value_str) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        Some(AttributeKind::RustcReservationImpl(cx.attr_span, value_str))
    }
}

pub(crate) struct PreludeImportParser;

impl<S: Stage> NoArgsAttributeParser<S> for PreludeImportParser {
    const PATH: &[Symbol] = &[sym::prelude_import];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Use)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::PreludeImport;
}

pub(crate) struct RustcDocPrimitiveParser;

impl<S: Stage> SingleAttributeParser<S> for RustcDocPrimitiveParser {
    const PATH: &[Symbol] = &[sym::rustc_doc_primitive];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Mod)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "primitive name");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(args.span().unwrap_or(cx.attr_span), None);
            return None;
        };

        let Some(value_str) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        Some(AttributeKind::RustcDocPrimitive(cx.attr_span, value_str))
    }
}

pub(crate) struct RustcIntrinsicParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcIntrinsicParser {
    const PATH: &[Symbol] = &[sym::rustc_intrinsic];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcIntrinsic;
}

pub(crate) struct RustcIntrinsicConstStableIndirectParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcIntrinsicConstStableIndirectParser {
    const PATH: &'static [Symbol] = &[sym::rustc_intrinsic_const_stable_indirect];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcIntrinsicConstStableIndirect;
}
