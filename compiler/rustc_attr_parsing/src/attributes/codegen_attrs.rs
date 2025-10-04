use rustc_hir::attrs::{CoverageAttrKind, OptimizeAttr, SanitizerSet, UsedBy};
use rustc_session::parse::feature_err;

use super::prelude::*;
use crate::session_diagnostics::{
    NakedFunctionIncompatibleAttribute, NullOnExport, NullOnObjcClass, NullOnObjcSelector,
    ObjcClassExpectedStringLiteral, ObjcSelectorExpectedStringLiteral,
};

pub(crate) struct OptimizeParser;

impl<S: Stage> SingleAttributeParser<S> for OptimizeParser {
    const PATH: &[Symbol] = &[sym::optimize];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Closure),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Inherent)),
    ]);
    const TEMPLATE: AttributeTemplate = template!(List: &["size", "speed", "none"]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span);
            return None;
        };

        let Some(single) = list.single() else {
            cx.expected_single_argument(list.span);
            return None;
        };

        let res = match single.meta_item().and_then(|i| i.path().word().map(|i| i.name)) {
            Some(sym::size) => OptimizeAttr::Size,
            Some(sym::speed) => OptimizeAttr::Speed,
            Some(sym::none) => OptimizeAttr::DoNotOptimize,
            _ => {
                cx.expected_specific_argument(single.span(), &[sym::size, sym::speed, sym::none]);
                OptimizeAttr::Default
            }
        };

        Some(AttributeKind::Optimize(res, cx.attr_span))
    }
}

pub(crate) struct ColdParser;

impl<S: Stage> NoArgsAttributeParser<S> for ColdParser {
    const PATH: &[Symbol] = &[sym::cold];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::ForeignFn),
        Allow(Target::Closure),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::Cold;
}

pub(crate) struct CoverageParser;

impl<S: Stage> SingleAttributeParser<S> for CoverageParser {
    const PATH: &[Symbol] = &[sym::coverage];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Closure),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Impl { of_trait: true }),
        Allow(Target::Impl { of_trait: false }),
        Allow(Target::Mod),
        Allow(Target::Crate),
    ]);
    const TEMPLATE: AttributeTemplate = template!(OneOf: &[sym::off, sym::on]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(args) = args.list() else {
            cx.expected_specific_argument_and_list(cx.attr_span, &[sym::on, sym::off]);
            return None;
        };

        let Some(arg) = args.single() else {
            cx.expected_single_argument(args.span);
            return None;
        };

        let fail_incorrect_argument =
            |span| cx.expected_specific_argument(span, &[sym::on, sym::off]);

        let Some(arg) = arg.meta_item() else {
            fail_incorrect_argument(args.span);
            return None;
        };

        let kind = match arg.path().word_sym() {
            Some(sym::off) => CoverageAttrKind::Off,
            Some(sym::on) => CoverageAttrKind::On,
            None | Some(_) => {
                fail_incorrect_argument(arg.span());
                return None;
            }
        };

        Some(AttributeKind::Coverage(cx.attr_span, kind))
    }
}

pub(crate) struct ExportNameParser;

impl<S: Stage> SingleAttributeParser<S> for ExportNameParser {
    const PATH: &[rustc_span::Symbol] = &[sym::export_name];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Static),
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
        Warn(Target::MacroCall),
    ]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "name");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        let Some(name) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };
        if name.as_str().contains('\0') {
            // `#[export_name = ...]` will be converted to a null-terminated string,
            // so it may not contain any null characters.
            cx.emit_err(NullOnExport { span: cx.attr_span });
            return None;
        }
        Some(AttributeKind::ExportName { name, span: cx.attr_span })
    }
}

pub(crate) struct ObjcClassParser;

impl<S: Stage> SingleAttributeParser<S> for ObjcClassParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_objc_class];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::ForeignStatic)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "ClassName");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        let Some(classname) = nv.value_as_str() else {
            // `#[rustc_objc_class = ...]` is expected to be used as an implementatioin detail
            // inside a standard library macro, but `cx.expected_string_literal` exposes too much.
            // Use a custom error message instead.
            cx.emit_err(ObjcClassExpectedStringLiteral { span: nv.value_span });
            return None;
        };
        if classname.as_str().contains('\0') {
            // `#[rustc_objc_class = ...]` will be converted to a null-terminated string,
            // so it may not contain any null characters.
            cx.emit_err(NullOnObjcClass { span: nv.value_span });
            return None;
        }
        Some(AttributeKind::ObjcClass { classname, span: cx.attr_span })
    }
}

pub(crate) struct ObjcSelectorParser;

impl<S: Stage> SingleAttributeParser<S> for ObjcSelectorParser {
    const PATH: &[rustc_span::Symbol] = &[sym::rustc_objc_selector];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::ForeignStatic)]);
    const TEMPLATE: AttributeTemplate = template!(NameValueStr: "methodName");

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        let Some(methname) = nv.value_as_str() else {
            // `#[rustc_objc_selector = ...]` is expected to be used as an implementatioin detail
            // inside a standard library macro, but `cx.expected_string_literal` exposes too much.
            // Use a custom error message instead.
            cx.emit_err(ObjcSelectorExpectedStringLiteral { span: nv.value_span });
            return None;
        };
        if methname.as_str().contains('\0') {
            // `#[rustc_objc_selector = ...]` will be converted to a null-terminated string,
            // so it may not contain any null characters.
            cx.emit_err(NullOnObjcSelector { span: nv.value_span });
            return None;
        }
        Some(AttributeKind::ObjcSelector { methname, span: cx.attr_span })
    }
}

#[derive(Default)]
pub(crate) struct NakedParser {
    span: Option<Span>,
}

impl<S: Stage> AttributeParser<S> for NakedParser {
    const ATTRIBUTES: AcceptMapping<Self, S> =
        &[(&[sym::naked], template!(Word), |this, cx, args| {
            if let Err(span) = args.no_args() {
                cx.expected_no_args(span);
                return;
            }

            if let Some(earlier) = this.span {
                let span = cx.attr_span;
                cx.warn_unused_duplicate(earlier, span);
            } else {
                this.span = Some(cx.attr_span);
            }
        })];
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Warn(Target::MacroCall),
    ]);

    fn finalize(self, cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        // FIXME(jdonszelmann): upgrade this list to *parsed* attributes
        // once all of these have parsed forms. That'd make the check much nicer...
        //
        // many attributes don't make sense in combination with #[naked].
        // Notable attributes that are incompatible with `#[naked]` are:
        //
        // * `#[inline]`
        // * `#[track_caller]`
        // * `#[test]`, `#[ignore]`, `#[should_panic]`
        //
        // NOTE: when making changes to this list, check that `error_codes/E0736.md` remains
        // accurate.
        const ALLOW_LIST: &[rustc_span::Symbol] = &[
            // conditional compilation
            sym::cfg_trace,
            sym::cfg_attr_trace,
            // testing (allowed here so better errors can be generated in `rustc_builtin_macros::test`)
            sym::test,
            sym::ignore,
            sym::should_panic,
            sym::bench,
            // diagnostics
            sym::allow,
            sym::warn,
            sym::deny,
            sym::forbid,
            sym::deprecated,
            sym::must_use,
            // abi, linking and FFI
            sym::cold,
            sym::export_name,
            sym::link_section,
            sym::linkage,
            sym::no_mangle,
            sym::instruction_set,
            sym::repr,
            sym::rustc_std_internal_symbol,
            // FIXME(#82232, #143834): temporarily renamed to mitigate `#[align]` nameres ambiguity
            sym::rustc_align,
            sym::rustc_align_static,
            // obviously compatible with self
            sym::naked,
            // documentation
            sym::doc,
        ];

        let span = self.span?;

        // only if we found a naked attribute do we do the somewhat expensive check
        'outer: for other_attr in cx.all_attrs {
            for allowed_attr in ALLOW_LIST {
                if other_attr.segments().next().is_some_and(|i| cx.tools.contains(&i.name)) {
                    // effectively skips the error message  being emitted below
                    // if it's a tool attribute
                    continue 'outer;
                }
                if other_attr.word_is(*allowed_attr) {
                    // effectively skips the error message  being emitted below
                    // if its an allowed attribute
                    continue 'outer;
                }

                if other_attr.word_is(sym::target_feature) {
                    if !cx.features().naked_functions_target_feature() {
                        feature_err(
                            &cx.sess(),
                            sym::naked_functions_target_feature,
                            other_attr.span(),
                            "`#[target_feature(/* ... */)]` is currently unstable on `#[naked]` functions",
                        ).emit();
                    }

                    continue 'outer;
                }
            }

            cx.emit_err(NakedFunctionIncompatibleAttribute {
                span: other_attr.span(),
                naked_span: span,
                attr: other_attr.get_attribute_path().to_string(),
            });
        }

        Some(AttributeKind::Naked(span))
    }
}

pub(crate) struct TrackCallerParser;
impl<S: Stage> NoArgsAttributeParser<S> for TrackCallerParser {
    const PATH: &[Symbol] = &[sym::track_caller];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::ForeignFn),
        Allow(Target::Closure),
        Warn(Target::MacroDef),
        Warn(Target::Arm),
        Warn(Target::Field),
        Warn(Target::MacroCall),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::TrackCaller;
}

pub(crate) struct NoMangleParser;
impl<S: Stage> NoArgsAttributeParser<S> for NoMangleParser {
    const PATH: &[Symbol] = &[sym::no_mangle];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::Fn),
        Allow(Target::Static),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::NoMangle;
}

#[derive(Default)]
pub(crate) struct UsedParser {
    first_compiler: Option<Span>,
    first_linker: Option<Span>,
    first_default: Option<Span>,
}

// A custom `AttributeParser` is used rather than a Simple attribute parser because
// - Specifying two `#[used]` attributes is a warning (but will be an error in the future)
// - But specifying two conflicting attributes: `#[used(compiler)]` and `#[used(linker)]` is already an error today
// We can change this to a Simple parser once the warning becomes an error
impl<S: Stage> AttributeParser<S> for UsedParser {
    const ATTRIBUTES: AcceptMapping<Self, S> = &[(
        &[sym::used],
        template!(Word, List: &["compiler", "linker"]),
        |group: &mut Self, cx, args| {
            let used_by = match args {
                ArgParser::NoArgs => UsedBy::Default,
                ArgParser::List(list) => {
                    let Some(l) = list.single() else {
                        cx.expected_single_argument(list.span);
                        return;
                    };

                    match l.meta_item().and_then(|i| i.path().word_sym()) {
                        Some(sym::compiler) => {
                            if !cx.features().used_with_arg() {
                                feature_err(
                                    &cx.sess(),
                                    sym::used_with_arg,
                                    cx.attr_span,
                                    "`#[used(compiler)]` is currently unstable",
                                )
                                .emit();
                            }
                            UsedBy::Compiler
                        }
                        Some(sym::linker) => {
                            if !cx.features().used_with_arg() {
                                feature_err(
                                    &cx.sess(),
                                    sym::used_with_arg,
                                    cx.attr_span,
                                    "`#[used(linker)]` is currently unstable",
                                )
                                .emit();
                            }
                            UsedBy::Linker
                        }
                        _ => {
                            cx.expected_specific_argument(l.span(), &[sym::compiler, sym::linker]);
                            return;
                        }
                    }
                }
                ArgParser::NameValue(_) => return,
            };

            let attr_span = cx.attr_span;

            // `#[used]` is interpreted as `#[used(linker)]` (though depending on target OS the
            // circumstances are more complicated). While we're checking `used_by`, also report
            // these cross-`UsedBy` duplicates to warn.
            let target = match used_by {
                UsedBy::Compiler => &mut group.first_compiler,
                UsedBy::Linker => {
                    if let Some(prev) = group.first_default {
                        cx.warn_unused_duplicate(prev, attr_span);
                        return;
                    }
                    &mut group.first_linker
                }
                UsedBy::Default => {
                    if let Some(prev) = group.first_linker {
                        cx.warn_unused_duplicate(prev, attr_span);
                        return;
                    }
                    &mut group.first_default
                }
            };

            if let Some(prev) = *target {
                cx.warn_unused_duplicate(prev, attr_span);
            } else {
                *target = Some(attr_span);
            }
        },
    )];
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowList(&[Allow(Target::Static), Warn(Target::MacroCall)]);

    fn finalize(self, _cx: &FinalizeContext<'_, '_, S>) -> Option<AttributeKind> {
        // If a specific form of `used` is specified, it takes precedence over generic `#[used]`.
        // If both `linker` and `compiler` are specified, use `linker`.
        Some(match (self.first_compiler, self.first_linker, self.first_default) {
            (_, Some(span), _) => AttributeKind::Used { used_by: UsedBy::Linker, span },
            (Some(span), _, _) => AttributeKind::Used { used_by: UsedBy::Compiler, span },
            (_, _, Some(span)) => AttributeKind::Used { used_by: UsedBy::Default, span },
            (None, None, None) => return None,
        })
    }
}

fn parse_tf_attribute<'c, S: Stage>(
    cx: &'c mut AcceptContext<'_, '_, S>,
    args: &'c ArgParser<'_>,
) -> impl IntoIterator<Item = (Symbol, Span)> + 'c {
    let mut features = Vec::new();
    let ArgParser::List(list) = args else {
        cx.expected_list(cx.attr_span);
        return features;
    };
    if list.is_empty() {
        cx.warn_empty_attribute(cx.attr_span);
        return features;
    }
    for item in list.mixed() {
        let Some(name_value) = item.meta_item() else {
            cx.expected_name_value(item.span(), Some(sym::enable));
            return features;
        };

        // Validate name
        let Some(name) = name_value.path().word_sym() else {
            cx.expected_name_value(name_value.path().span(), Some(sym::enable));
            return features;
        };
        if name != sym::enable {
            cx.expected_name_value(name_value.path().span(), Some(sym::enable));
            return features;
        }

        // Use value
        let Some(name_value) = name_value.args().name_value() else {
            cx.expected_name_value(item.span(), Some(sym::enable));
            return features;
        };
        let Some(value_str) = name_value.value_as_str() else {
            cx.expected_string_literal(name_value.value_span, Some(name_value.value_as_lit()));
            return features;
        };
        for feature in value_str.as_str().split(",") {
            features.push((Symbol::intern(feature), item.span()));
        }
    }
    features
}

pub(crate) struct TargetFeatureParser;

impl<S: Stage> CombineAttributeParser<S> for TargetFeatureParser {
    type Item = (Symbol, Span);
    const PATH: &[Symbol] = &[sym::target_feature];
    const CONVERT: ConvertFn<Self::Item> = |items, span| AttributeKind::TargetFeature {
        features: items,
        attr_span: span,
        was_forced: false,
    };
    const TEMPLATE: AttributeTemplate = template!(List: &["enable = \"feat1, feat2\""]);

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        parse_tf_attribute(cx, args)
    }

    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Warn(Target::Statement),
        Warn(Target::Field),
        Warn(Target::Arm),
        Warn(Target::MacroDef),
        Warn(Target::MacroCall),
    ]);
}

pub(crate) struct ForceTargetFeatureParser;

impl<S: Stage> CombineAttributeParser<S> for ForceTargetFeatureParser {
    type Item = (Symbol, Span);
    const PATH: &[Symbol] = &[sym::force_target_feature];
    const CONVERT: ConvertFn<Self::Item> = |items, span| AttributeKind::TargetFeature {
        features: items,
        attr_span: span,
        was_forced: true,
    };
    const TEMPLATE: AttributeTemplate = template!(List: &["enable = \"feat1, feat2\""]);
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
    ]);

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        parse_tf_attribute(cx, args)
    }
}

pub(crate) struct SanitizeParser;

impl<S: Stage> SingleAttributeParser<S> for SanitizeParser {
    const PATH: &[Symbol] = &[sym::sanitize];

    // FIXME: still checked in check_attrs.rs
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS);

    const TEMPLATE: AttributeTemplate = template!(List: &[
        r#"address = "on|off""#,
        r#"kernel_address = "on|off""#,
        r#"cfi = "on|off""#,
        r#"hwaddress = "on|off""#,
        r#"kcfi = "on|off""#,
        r#"memory = "on|off""#,
        r#"memtag = "on|off""#,
        r#"shadow_call_stack = "on|off""#,
        r#"thread = "on|off""#
    ]);

    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(list) = args.list() else {
            cx.expected_list(cx.attr_span);
            return None;
        };

        let mut on_set = SanitizerSet::empty();
        let mut off_set = SanitizerSet::empty();

        for item in list.mixed() {
            let Some(item) = item.meta_item() else {
                cx.expected_name_value(item.span(), None);
                continue;
            };

            let path = item.path().word_sym();
            let Some(value) = item.args().name_value() else {
                cx.expected_name_value(item.span(), path);
                continue;
            };

            let mut apply = |s: SanitizerSet| {
                let is_on = match value.value_as_str() {
                    Some(sym::on) => true,
                    Some(sym::off) => false,
                    Some(_) => {
                        cx.expected_specific_argument_strings(
                            value.value_span,
                            &[sym::on, sym::off],
                        );
                        return;
                    }
                    None => {
                        cx.expected_string_literal(value.value_span, Some(value.value_as_lit()));
                        return;
                    }
                };

                if is_on {
                    on_set |= s;
                } else {
                    off_set |= s;
                }
            };

            match path {
                Some(sym::address) | Some(sym::kernel_address) => {
                    apply(SanitizerSet::ADDRESS | SanitizerSet::KERNELADDRESS)
                }
                Some(sym::cfi) => apply(SanitizerSet::CFI),
                Some(sym::kcfi) => apply(SanitizerSet::KCFI),
                Some(sym::memory) => apply(SanitizerSet::MEMORY),
                Some(sym::memtag) => apply(SanitizerSet::MEMTAG),
                Some(sym::shadow_call_stack) => apply(SanitizerSet::SHADOWCALLSTACK),
                Some(sym::thread) => apply(SanitizerSet::THREAD),
                Some(sym::hwaddress) => apply(SanitizerSet::HWADDRESS),
                _ => {
                    cx.expected_specific_argument_strings(
                        item.path().span(),
                        &[
                            sym::address,
                            sym::cfi,
                            sym::kcfi,
                            sym::memory,
                            sym::memtag,
                            sym::shadow_call_stack,
                            sym::thread,
                            sym::hwaddress,
                        ],
                    );
                    continue;
                }
            }
        }

        Some(AttributeKind::Sanitize { on_set, off_set, span: cx.attr_span })
    }
}
