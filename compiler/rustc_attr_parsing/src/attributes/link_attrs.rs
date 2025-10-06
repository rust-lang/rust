use rustc_feature::Features;
use rustc_hir::attrs::AttributeKind::{LinkName, LinkOrdinal, LinkSection};
use rustc_hir::attrs::*;
use rustc_session::Session;
use rustc_session::parse::feature_err;
use rustc_span::kw;
use rustc_target::spec::BinaryFormat;

use super::prelude::*;
use super::util::parse_single_integer;
use crate::attributes::cfg::parse_cfg_entry;
use crate::fluent_generated;
use crate::session_diagnostics::{
    AsNeededCompatibility, BundleNeedsStatic, EmptyLinkName, ImportNameTypeRaw, ImportNameTypeX86,
    IncompatibleWasmLink, InvalidLinkModifier, LinkFrameworkApple, LinkOrdinalOutOfRange,
    LinkRequiresName, MultipleModifiers, NullOnLinkSection, RawDylibNoNul, RawDylibOnlyWindows,
    WholeArchiveNeedsStatic,
};

pub(crate) struct LinkNameParser;

impl<S: Stage> SingleAttributeParser<S> for LinkNameParser {
    const PATH: &[Symbol] = &[sym::link_name];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowListWarnRest(&[
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        NameValueStr: "name",
        "https://doc.rust-lang.org/reference/items/external-blocks.html#the-link_name-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(nv) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, None);
            return None;
        };
        let Some(name) = nv.value_as_str() else {
            cx.expected_string_literal(nv.value_span, Some(nv.value_as_lit()));
            return None;
        };

        Some(LinkName { name, span: cx.attr_span })
    }
}

pub(crate) struct LinkParser;

impl<S: Stage> CombineAttributeParser<S> for LinkParser {
    type Item = LinkEntry;
    const PATH: &[Symbol] = &[sym::link];
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::Link;
    const TEMPLATE: AttributeTemplate = template!(List: &[
            r#"name = "...""#,
            r#"name = "...", kind = "dylib|static|...""#,
            r#"name = "...", wasm_import_module = "...""#,
            r#"name = "...", import_name_type = "decorated|noprefix|undecorated""#,
            r#"name = "...", kind = "dylib|static|...", wasm_import_module = "...", import_name_type = "decorated|noprefix|undecorated""#,
        ], "https://doc.rust-lang.org/reference/items/external-blocks.html#the-link-attribute");
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS); //FIXME Still checked fully in `check_attr.rs`

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        let items = match args {
            ArgParser::List(list) => list,
            // This is an edgecase added because making this a hard error would break too many crates
            // Specifically `#[link = "dl"]` is accepted with a FCW
            // For more information, see https://github.com/rust-lang/rust/pull/143193
            ArgParser::NameValue(nv) if nv.value_as_str().is_some_and(|v| v == sym::dl) => {
                let suggestions = <Self as CombineAttributeParser<S>>::TEMPLATE
                    .suggestions(cx.attr_style, "link");
                let span = cx.attr_span;
                cx.emit_lint(AttributeLintKind::IllFormedAttributeInput { suggestions }, span);
                return None;
            }
            _ => {
                cx.expected_list(cx.attr_span);
                return None;
            }
        };

        let sess = cx.sess();
        let features = cx.features();

        let mut name = None;
        let mut kind = None;
        let mut modifiers = None;
        let mut cfg = None;
        let mut wasm_import_module = None;
        let mut import_name_type = None;
        for item in items.mixed() {
            let Some(item) = item.meta_item() else {
                cx.unexpected_literal(item.span());
                continue;
            };

            let cont = match item.path().word().map(|ident| ident.name) {
                Some(sym::name) => Self::parse_link_name(item, &mut name, cx),
                Some(sym::kind) => Self::parse_link_kind(item, &mut kind, cx, sess, features),
                Some(sym::modifiers) => Self::parse_link_modifiers(item, &mut modifiers, cx),
                Some(sym::cfg) => Self::parse_link_cfg(item, &mut cfg, cx, sess, features),
                Some(sym::wasm_import_module) => {
                    Self::parse_link_wasm_import_module(item, &mut wasm_import_module, cx)
                }
                Some(sym::import_name_type) => {
                    Self::parse_link_import_name_type(item, &mut import_name_type, cx)
                }
                _ => {
                    cx.expected_specific_argument_strings(
                        item.span(),
                        &[
                            sym::name,
                            sym::kind,
                            sym::modifiers,
                            sym::cfg,
                            sym::wasm_import_module,
                            sym::import_name_type,
                        ],
                    );
                    true
                }
            };
            if !cont {
                return None;
            }
        }

        // Do this outside the above loop so we don't depend on modifiers coming after kinds
        let mut verbatim = None;
        if let Some((modifiers, span)) = modifiers {
            for modifier in modifiers.as_str().split(',') {
                let (modifier, value): (Symbol, bool) = match modifier.strip_prefix(&['+', '-']) {
                    Some(m) => (Symbol::intern(m), modifier.starts_with('+')),
                    None => {
                        cx.emit_err(InvalidLinkModifier { span });
                        continue;
                    }
                };

                macro report_unstable_modifier($feature: ident) {
                    if !features.$feature() {
                        // FIXME: make this translatable
                        #[expect(rustc::untranslatable_diagnostic)]
                        feature_err(
                            sess,
                            sym::$feature,
                            span,
                            format!("linking modifier `{modifier}` is unstable"),
                        )
                        .emit();
                    }
                }
                let assign_modifier = |dst: &mut Option<bool>| {
                    if dst.is_some() {
                        cx.emit_err(MultipleModifiers { span, modifier });
                    } else {
                        *dst = Some(value);
                    }
                };
                match (modifier, &mut kind) {
                    (sym::bundle, Some(NativeLibKind::Static { bundle, .. })) => {
                        assign_modifier(bundle)
                    }
                    (sym::bundle, _) => {
                        cx.emit_err(BundleNeedsStatic { span });
                    }

                    (sym::verbatim, _) => assign_modifier(&mut verbatim),

                    (
                        sym::whole_dash_archive,
                        Some(NativeLibKind::Static { whole_archive, .. }),
                    ) => assign_modifier(whole_archive),
                    (sym::whole_dash_archive, _) => {
                        cx.emit_err(WholeArchiveNeedsStatic { span });
                    }

                    (sym::as_dash_needed, Some(NativeLibKind::Dylib { as_needed }))
                    | (sym::as_dash_needed, Some(NativeLibKind::Framework { as_needed }))
                    | (sym::as_dash_needed, Some(NativeLibKind::RawDylib { as_needed })) => {
                        report_unstable_modifier!(native_link_modifiers_as_needed);
                        assign_modifier(as_needed)
                    }
                    (sym::as_dash_needed, _) => {
                        cx.emit_err(AsNeededCompatibility { span });
                    }

                    _ => {
                        cx.expected_specific_argument_strings(
                            span,
                            &[
                                sym::bundle,
                                sym::verbatim,
                                sym::whole_dash_archive,
                                sym::as_dash_needed,
                            ],
                        );
                    }
                }
            }
        }

        if let Some((_, span)) = wasm_import_module {
            if name.is_some() || kind.is_some() || modifiers.is_some() || cfg.is_some() {
                cx.emit_err(IncompatibleWasmLink { span });
            }
        }

        if wasm_import_module.is_some() {
            (name, kind) = (wasm_import_module, Some(NativeLibKind::WasmImportModule));
        }
        let Some((name, name_span)) = name else {
            cx.emit_err(LinkRequiresName { span: cx.attr_span });
            return None;
        };

        // Do this outside of the loop so that `import_name_type` can be specified before `kind`.
        if let Some((_, span)) = import_name_type {
            if !matches!(kind, Some(NativeLibKind::RawDylib { .. })) {
                cx.emit_err(ImportNameTypeRaw { span });
            }
        }

        if let Some(NativeLibKind::RawDylib { .. }) = kind
            && name.as_str().contains('\0')
        {
            cx.emit_err(RawDylibNoNul { span: name_span });
        }

        Some(LinkEntry {
            span: cx.attr_span,
            kind: kind.unwrap_or(NativeLibKind::Unspecified),
            name,
            cfg,
            verbatim,
            import_name_type,
        })
    }
}

impl LinkParser {
    fn parse_link_name<S: Stage>(
        item: &MetaItemParser<'_>,
        name: &mut Option<(Symbol, Span)>,
        cx: &mut AcceptContext<'_, '_, S>,
    ) -> bool {
        if name.is_some() {
            cx.duplicate_key(item.span(), sym::name);
            return true;
        }
        let Some(nv) = item.args().name_value() else {
            cx.expected_name_value(item.span(), Some(sym::name));
            return false;
        };
        let Some(link_name) = nv.value_as_str() else {
            cx.expected_name_value(item.span(), Some(sym::name));
            return false;
        };

        if link_name.is_empty() {
            cx.emit_err(EmptyLinkName { span: nv.value_span });
        }
        *name = Some((link_name, nv.value_span));
        true
    }

    fn parse_link_kind<S: Stage>(
        item: &MetaItemParser<'_>,
        kind: &mut Option<NativeLibKind>,
        cx: &mut AcceptContext<'_, '_, S>,
        sess: &Session,
        features: &Features,
    ) -> bool {
        if kind.is_some() {
            cx.duplicate_key(item.span(), sym::kind);
            return true;
        }
        let Some(nv) = item.args().name_value() else {
            cx.expected_name_value(item.span(), Some(sym::kind));
            return true;
        };
        let Some(link_kind) = nv.value_as_str() else {
            cx.expected_name_value(item.span(), Some(sym::kind));
            return true;
        };

        let link_kind = match link_kind {
            kw::Static => NativeLibKind::Static { bundle: None, whole_archive: None },
            sym::dylib => NativeLibKind::Dylib { as_needed: None },
            sym::framework => {
                if !sess.target.is_like_darwin {
                    cx.emit_err(LinkFrameworkApple { span: nv.value_span });
                }
                NativeLibKind::Framework { as_needed: None }
            }
            sym::raw_dash_dylib => {
                if sess.target.is_like_windows {
                    // raw-dylib is stable and working on Windows
                } else if sess.target.binary_format == BinaryFormat::Elf && features.raw_dylib_elf()
                {
                    // raw-dylib is unstable on ELF, but the user opted in
                } else if sess.target.binary_format == BinaryFormat::Elf && sess.is_nightly_build()
                {
                    feature_err(
                        sess,
                        sym::raw_dylib_elf,
                        nv.value_span,
                        fluent_generated::attr_parsing_raw_dylib_elf_unstable,
                    )
                    .emit();
                } else {
                    cx.emit_err(RawDylibOnlyWindows { span: nv.value_span });
                }

                NativeLibKind::RawDylib { as_needed: None }
            }
            sym::link_dash_arg => {
                if !features.link_arg_attribute() {
                    feature_err(
                        sess,
                        sym::link_arg_attribute,
                        nv.value_span,
                        fluent_generated::attr_parsing_link_arg_unstable,
                    )
                    .emit();
                }
                NativeLibKind::LinkArg
            }
            _kind => {
                cx.expected_specific_argument_strings(
                    nv.value_span,
                    &[
                        kw::Static,
                        sym::dylib,
                        sym::framework,
                        sym::raw_dash_dylib,
                        sym::link_dash_arg,
                    ],
                );
                return true;
            }
        };
        *kind = Some(link_kind);
        true
    }

    fn parse_link_modifiers<S: Stage>(
        item: &MetaItemParser<'_>,
        modifiers: &mut Option<(Symbol, Span)>,
        cx: &mut AcceptContext<'_, '_, S>,
    ) -> bool {
        if modifiers.is_some() {
            cx.duplicate_key(item.span(), sym::modifiers);
            return true;
        }
        let Some(nv) = item.args().name_value() else {
            cx.expected_name_value(item.span(), Some(sym::modifiers));
            return true;
        };
        let Some(link_modifiers) = nv.value_as_str() else {
            cx.expected_name_value(item.span(), Some(sym::modifiers));
            return true;
        };
        *modifiers = Some((link_modifiers, nv.value_span));
        true
    }

    fn parse_link_cfg<S: Stage>(
        item: &MetaItemParser<'_>,
        cfg: &mut Option<CfgEntry>,
        cx: &mut AcceptContext<'_, '_, S>,
        sess: &Session,
        features: &Features,
    ) -> bool {
        if cfg.is_some() {
            cx.duplicate_key(item.span(), sym::cfg);
            return true;
        }
        let Some(link_cfg) = item.args().list() else {
            cx.expected_list(item.span());
            return true;
        };
        let Some(link_cfg) = link_cfg.single() else {
            cx.expected_single_argument(item.span());
            return true;
        };
        if !features.link_cfg() {
            feature_err(
                sess,
                sym::link_cfg,
                item.span(),
                fluent_generated::attr_parsing_link_cfg_unstable,
            )
            .emit();
        }
        *cfg = parse_cfg_entry(cx, link_cfg);
        true
    }

    fn parse_link_wasm_import_module<S: Stage>(
        item: &MetaItemParser<'_>,
        wasm_import_module: &mut Option<(Symbol, Span)>,
        cx: &mut AcceptContext<'_, '_, S>,
    ) -> bool {
        if wasm_import_module.is_some() {
            cx.duplicate_key(item.span(), sym::wasm_import_module);
            return true;
        }
        let Some(nv) = item.args().name_value() else {
            cx.expected_name_value(item.span(), Some(sym::wasm_import_module));
            return true;
        };
        let Some(link_wasm_import_module) = nv.value_as_str() else {
            cx.expected_name_value(item.span(), Some(sym::wasm_import_module));
            return true;
        };
        *wasm_import_module = Some((link_wasm_import_module, item.span()));
        true
    }

    fn parse_link_import_name_type<S: Stage>(
        item: &MetaItemParser<'_>,
        import_name_type: &mut Option<(PeImportNameType, Span)>,
        cx: &mut AcceptContext<'_, '_, S>,
    ) -> bool {
        if import_name_type.is_some() {
            cx.duplicate_key(item.span(), sym::import_name_type);
            return true;
        }
        let Some(nv) = item.args().name_value() else {
            cx.expected_name_value(item.span(), Some(sym::import_name_type));
            return true;
        };
        let Some(link_import_name_type) = nv.value_as_str() else {
            cx.expected_name_value(item.span(), Some(sym::import_name_type));
            return true;
        };
        if cx.sess().target.arch != "x86" {
            cx.emit_err(ImportNameTypeX86 { span: item.span() });
            return true;
        }

        let link_import_name_type = match link_import_name_type {
            sym::decorated => PeImportNameType::Decorated,
            sym::noprefix => PeImportNameType::NoPrefix,
            sym::undecorated => PeImportNameType::Undecorated,
            _ => {
                cx.expected_specific_argument_strings(
                    item.span(),
                    &[sym::decorated, sym::noprefix, sym::undecorated],
                );
                return true;
            }
        };
        *import_name_type = Some((link_import_name_type, item.span()));
        true
    }
}

pub(crate) struct LinkSectionParser;

impl<S: Stage> SingleAttributeParser<S> for LinkSectionParser {
    const PATH: &[Symbol] = &[sym::link_section];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
    const ALLOWED_TARGETS: AllowedTargets =
        AllowedTargets::AllowListWarnRest(&[Allow(Target::Static), Allow(Target::Fn)]);
    const TEMPLATE: AttributeTemplate = template!(
        NameValueStr: "name",
        "https://doc.rust-lang.org/reference/abi.html#the-link_section-attribute"
    );

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
            // `#[link_section = ...]` will be converted to a null-terminated string,
            // so it may not contain any null characters.
            cx.emit_err(NullOnLinkSection { span: cx.attr_span });
            return None;
        }

        Some(LinkSection { name, span: cx.attr_span })
    }
}

pub(crate) struct ExportStableParser;
impl<S: Stage> NoArgsAttributeParser<S> for ExportStableParser {
    const PATH: &[Symbol] = &[sym::export_stable];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(ALL_TARGETS); //FIXME Still checked fully in `check_attr.rs`
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::ExportStable;
}

pub(crate) struct FfiConstParser;
impl<S: Stage> NoArgsAttributeParser<S> for FfiConstParser {
    const PATH: &[Symbol] = &[sym::ffi_const];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::FfiConst;
}

pub(crate) struct FfiPureParser;
impl<S: Stage> NoArgsAttributeParser<S> for FfiPureParser {
    const PATH: &[Symbol] = &[sym::ffi_pure];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::ForeignFn)]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::FfiPure;
}

pub(crate) struct StdInternalSymbolParser;
impl<S: Stage> NoArgsAttributeParser<S> for StdInternalSymbolParser {
    const PATH: &[Symbol] = &[sym::rustc_std_internal_symbol];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::ForeignFn),
        Allow(Target::Static),
        Allow(Target::ForeignStatic),
    ]);
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::StdInternalSymbol;
}

pub(crate) struct LinkOrdinalParser;

impl<S: Stage> SingleAttributeParser<S> for LinkOrdinalParser {
    const PATH: &[Symbol] = &[sym::link_ordinal];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::ForeignFn),
        Allow(Target::ForeignStatic),
        Warn(Target::MacroCall),
    ]);
    const TEMPLATE: AttributeTemplate = template!(
        List: &["ordinal"],
        "https://doc.rust-lang.org/reference/items/external-blocks.html#the-link_ordinal-attribute"
    );

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let ordinal = parse_single_integer(cx, args)?;

        // According to the table at
        // https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#import-header, the
        // ordinal must fit into 16 bits. Similarly, the Ordinal field in COFFShortExport (defined
        // in llvm/include/llvm/Object/COFFImportFile.h), which we use to communicate import
        // information to LLVM for `#[link(kind = "raw-dylib"_])`, is also defined to be uint16_t.
        //
        // FIXME: should we allow an ordinal of 0?  The MSVC toolchain has inconsistent support for
        // this: both LINK.EXE and LIB.EXE signal errors and abort when given a .DEF file that
        // specifies a zero ordinal. However, llvm-dlltool is perfectly happy to generate an import
        // library for such a .DEF file, and MSVC's LINK.EXE is also perfectly happy to consume an
        // import library produced by LLVM with an ordinal of 0, and it generates an .EXE.  (I
        // don't know yet if the resulting EXE runs, as I haven't yet built the necessary DLL --
        // see earlier comment about LINK.EXE failing.)
        let Ok(ordinal) = ordinal.try_into() else {
            cx.emit_err(LinkOrdinalOutOfRange { span: cx.attr_span, ordinal });
            return None;
        };

        Some(LinkOrdinal { ordinal, span: cx.attr_span })
    }
}

pub(crate) struct LinkageParser;

impl<S: Stage> SingleAttributeParser<S> for LinkageParser {
    const PATH: &[Symbol] = &[sym::linkage];

    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;

    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[
        Allow(Target::Fn),
        Allow(Target::Method(MethodKind::Inherent)),
        Allow(Target::Method(MethodKind::Trait { body: false })),
        Allow(Target::Method(MethodKind::Trait { body: true })),
        Allow(Target::Method(MethodKind::TraitImpl)),
        Allow(Target::Static),
        Allow(Target::ForeignStatic),
        Allow(Target::ForeignFn),
    ]);

    const TEMPLATE: AttributeTemplate = template!(NameValueStr: [
        "available_externally",
        "common",
        "extern_weak",
        "external",
        "internal",
        "linkonce",
        "linkonce_odr",
        "weak",
        "weak_odr",
    ]);

    fn convert(cx: &mut AcceptContext<'_, '_, S>, args: &ArgParser<'_>) -> Option<AttributeKind> {
        let Some(name_value) = args.name_value() else {
            cx.expected_name_value(cx.attr_span, Some(sym::linkage));
            return None;
        };

        let Some(value) = name_value.value_as_str() else {
            cx.expected_string_literal(name_value.value_span, Some(name_value.value_as_lit()));
            return None;
        };

        // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
        // applicable to variable declarations and may not really make sense for
        // Rust code in the first place but allow them anyway and trust that the
        // user knows what they're doing. Who knows, unanticipated use cases may pop
        // up in the future.
        //
        // ghost, dllimport, dllexport and linkonce_odr_autohide are not supported
        // and don't have to be, LLVM treats them as no-ops.
        let linkage = match value {
            sym::available_externally => Linkage::AvailableExternally,
            sym::common => Linkage::Common,
            sym::extern_weak => Linkage::ExternalWeak,
            sym::external => Linkage::External,
            sym::internal => Linkage::Internal,
            sym::linkonce => Linkage::LinkOnceAny,
            sym::linkonce_odr => Linkage::LinkOnceODR,
            sym::weak => Linkage::WeakAny,
            sym::weak_odr => Linkage::WeakODR,

            _ => {
                cx.expected_specific_argument(
                    name_value.value_span,
                    &[
                        sym::available_externally,
                        sym::common,
                        sym::extern_weak,
                        sym::external,
                        sym::internal,
                        sym::linkonce,
                        sym::linkonce_odr,
                        sym::weak,
                        sym::weak_odr,
                    ],
                );
                return None;
            }
        };

        Some(AttributeKind::Linkage(linkage, cx.attr_span))
    }
}
