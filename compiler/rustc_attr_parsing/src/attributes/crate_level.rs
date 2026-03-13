use rustc_hir::attrs::{CrateType, WindowsSubsystemKind};
use rustc_hir::lints::AttributeLintKind;
use rustc_session::lint::builtin::UNKNOWN_CRATE_TYPES;
use rustc_span::Symbol;
use rustc_span::edit_distance::find_best_match_for_name;

use super::prelude::*;

pub(crate) struct CrateNameParser;

impl<S: Stage> SingleAttributeParser<S> for CrateNameParser {
    const PATH: &[Symbol] = &[sym::crate_name];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "name");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::NameValue(n) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        let Some(name) = n.value_as_str() else {
            cx.expected_string_literal(n.value_span, Some(n.value_as_lit()));
            return None;
        };

        Some(AttributeKind::CrateName { name, name_span: n.value_span, attr_span: cx.attr_span })
    }
}

pub(crate) struct CrateTypeParser;

impl<S: Stage> CombineAttributeParser<S> for CrateTypeParser {
    const PATH: &[Symbol] = &[sym::crate_type];
    type Item = CrateType;
    const CONVERT: ConvertFn<Self::Item> = |items, _| AttributeKind::CrateType(items);

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    const TEMPLATE: AttributeTemplate =
        template!(NameValueStr: "crate type", "https://doc.rust-lang.org/reference/linkage.html");

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let ArgParser::NameValue(n) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        let Some(crate_type) = n.value_as_str() else {
            cx.expected_string_literal(n.value_span, Some(n.value_as_lit()));
            return None;
        };

        let Ok(crate_type) = crate_type.try_into() else {
            // We don't error on invalid `#![crate_type]` when not applied to a crate
            if cx.shared.target == Target::Crate {
                let candidate = find_best_match_for_name(
                    &CrateType::all_stable().iter().map(|(name, _)| *name).collect::<Vec<_>>(),
                    crate_type,
                    None,
                );
                cx.emit_lint(
                    UNKNOWN_CRATE_TYPES,
                    AttributeLintKind::CrateTypeUnknown {
                        span: n.value_span,
                        suggested: candidate,
                    },
                    n.value_span,
                );
            }
            return None;
        };

        Some(crate_type)
    }
}

pub(crate) struct RecursionLimitParser;

impl<S: Stage> SingleAttributeParser<S> for RecursionLimitParser {
    const PATH: &[Symbol] = &[sym::recursion_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N", "https://doc.rust-lang.org/reference/attributes/limits.html#the-recursion_limit-attribute");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::RecursionLimit {
            limit: cx.parse_limit_int(nv)?,
            attr_span: cx.attr_span,
            limit_span: nv.value_span,
        })
    }
}

pub(crate) struct MoveSizeLimitParser;

impl<S: Stage> SingleAttributeParser<S> for MoveSizeLimitParser {
    const PATH: &[Symbol] = &[sym::move_size_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::MoveSizeLimit {
            limit: cx.parse_limit_int(nv)?,
            attr_span: cx.attr_span,
            limit_span: nv.value_span,
        })
    }
}

pub(crate) struct TypeLengthLimitParser;

impl<S: Stage> SingleAttributeParser<S> for TypeLengthLimitParser {
    const PATH: &[Symbol] = &[sym::type_length_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::TypeLengthLimit {
            limit: cx.parse_limit_int(nv)?,
            attr_span: cx.attr_span,
            limit_span: nv.value_span,
        })
    }
}

pub(crate) struct PatternComplexityLimitParser;

impl<S: Stage> SingleAttributeParser<S> for PatternComplexityLimitParser {
    const PATH: &[Symbol] = &[sym::pattern_complexity_limit];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "N");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let ArgParser::NameValue(nv) = args else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };

        Some(AttributeKind::PatternComplexityLimit {
            limit: cx.parse_limit_int(nv)?,
            attr_span: cx.attr_span,
            limit_span: nv.value_span,
        })
    }
}

pub(crate) struct NoCoreParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoCoreParser {
    const PATH: &[Symbol] = &[sym::no_core];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoCore;
}

pub(crate) struct NoStdParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoStdParser {
    const PATH: &[Symbol] = &[sym::no_std];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoStd;
}

pub(crate) struct NoMainParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoMainParser {
    const PATH: &[Symbol] = &[sym::no_main];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::NoMain;
}

pub(crate) struct RustcCoherenceIsCoreParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcCoherenceIsCoreParser {
    const PATH: &[Symbol] = &[sym::rustc_coherence_is_core];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::RustcCoherenceIsCore;
}

pub(crate) struct WindowsSubsystemParser;

impl<S: Stage> SingleAttributeParser<S> for WindowsSubsystemParser {
    const PATH: &[Symbol] = &[sym::windows_subsystem];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: ["windows", "console"], "https://doc.rust-lang.org/reference/runtime.html#the-windows_subsystem-attribute");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(
                args.span().unwrap_or(cx.inner_span),
                Some(sym::windows_subsystem),
            );
            return None;
        };

        let kind = match nv.value_as_str() {
            Some(sym::console) => WindowsSubsystemKind::Console,
            Some(sym::windows) => WindowsSubsystemKind::Windows,
            Some(_) | None => {
                cx.expected_specific_argument_strings(nv.value_span, &[sym::console, sym::windows]);
                return None;
            }
        };

        Some(AttributeKind::WindowsSubsystem(kind, cx.attr_span))
    }
}

pub(crate) struct PanicRuntimeParser;

impl<S: Stage> NoArgsAttributeParser<S> for PanicRuntimeParser {
    const PATH: &[Symbol] = &[sym::panic_runtime];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::PanicRuntime;
}

pub(crate) struct NeedsPanicRuntimeParser;

impl<S: Stage> NoArgsAttributeParser<S> for NeedsPanicRuntimeParser {
    const PATH: &[Symbol] = &[sym::needs_panic_runtime];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::NeedsPanicRuntime;
}

pub(crate) struct ProfilerRuntimeParser;

impl<S: Stage> NoArgsAttributeParser<S> for ProfilerRuntimeParser {
    const PATH: &[Symbol] = &[sym::profiler_runtime];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::ProfilerRuntime;
}

pub(crate) struct NoBuiltinsParser;

impl<S: Stage> NoArgsAttributeParser<S> for NoBuiltinsParser {
    const PATH: &[Symbol] = &[sym::no_builtins];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::NoBuiltins;
}

pub(crate) struct RustcPreserveUbChecksParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcPreserveUbChecksParser {
    const PATH: &[Symbol] = &[sym::rustc_preserve_ub_checks];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcPreserveUbChecks;
}

pub(crate) struct RustcNoImplicitBoundsParser;

impl<S: Stage> NoArgsAttributeParser<S> for RustcNoImplicitBoundsParser {
    const PATH: &[Symbol] = &[sym::rustc_no_implicit_bounds];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcNoImplicitBounds;
}

pub(crate) struct DefaultLibAllocatorParser;

impl<S: Stage> NoArgsAttributeParser<S> for DefaultLibAllocatorParser {
    const PATH: &[Symbol] = &[sym::default_lib_allocator];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::DefaultLibAllocator;
}

pub(crate) struct FeatureParser;

impl<S: Stage> CombineAttributeParser<S> for FeatureParser {
    const PATH: &[Symbol] = &[sym::feature];
    type Item = Ident;
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::Feature;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Crate)]);
    const TEMPLATE: AttributeTemplate = template!(List: &["feature1, feature2, ..."]);

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let ArgParser::List(list) = args else {
            cx.expected_list(cx.attr_span, args);
            return Vec::new();
        };

        if list.is_empty() {
            cx.warn_empty_attribute(cx.attr_span);
        }

        let mut res = Vec::new();

        for elem in list.mixed() {
            let Some(elem) = elem.meta_item() else {
                cx.expected_identifier(elem.span());
                continue;
            };
            if let Err(arg_span) = elem.args().no_args() {
                cx.expected_no_args(arg_span);
                continue;
            }

            let path = elem.path();
            let Some(ident) = path.word() else {
                cx.expected_identifier(path.span());
                continue;
            };
            res.push(ident);
        }

        res
    }
}

pub(crate) struct RegisterToolParser;

impl<S: Stage> CombineAttributeParser<S> for RegisterToolParser {
    const PATH: &[Symbol] = &[sym::register_tool];
    type Item = Ident;
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::RegisterTool;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);
    const TEMPLATE: AttributeTemplate = template!(List: &["tool1, tool2, ..."]);

    fn extend(
        cx: &mut AcceptContext<'_, '_, S>,
        args: &ArgParser,
    ) -> impl IntoIterator<Item = Self::Item> {
        let ArgParser::List(list) = args else {
            cx.expected_list(cx.attr_span, args);
            return Vec::new();
        };

        if list.is_empty() {
            cx.warn_empty_attribute(cx.attr_span);
        }

        let mut res = Vec::new();

        for elem in list.mixed() {
            let Some(elem) = elem.meta_item() else {
                cx.expected_identifier(elem.span());
                continue;
            };
            if let Err(arg_span) = elem.args().no_args() {
                cx.expected_no_args(arg_span);
                continue;
            }

            let path = elem.path();
            let Some(ident) = path.word() else {
                cx.expected_identifier(path.span());
                continue;
            };

            res.push(ident);
        }

        res
    }
}
