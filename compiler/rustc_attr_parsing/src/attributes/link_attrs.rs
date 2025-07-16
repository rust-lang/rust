use rustc_attr_data_structures::AttributeKind::{LinkName, LinkOrdinal, LinkSection};
use rustc_attr_data_structures::{AttributeKind, LinkEntry, NativeLibKind, PeImportNameType};
use rustc_feature::{AttributeTemplate, template};
use rustc_session::parse::feature_err;
use rustc_span::{Span, Symbol, sym};
use rustc_target::spec::BinaryFormat;

use crate::attributes::cfg::parse_cfg_entry;
use crate::attributes::{
    AttributeOrder, CombineAttributeParser, ConvertFn, NoArgsAttributeParser, OnDuplicate,
    SingleAttributeParser,
};
use crate::context::{AcceptContext, Stage, parse_single_integer};
use crate::fluent_generated;
use crate::parser::ArgParser;
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

        Some(LinkName { name, span: cx.attr_span })
    }
}

pub(crate) struct LinkParser;

impl<S: Stage> CombineAttributeParser<S> for LinkParser {
    type Item = LinkEntry;
    const PATH: &[Symbol] = &[sym::link];
    const CONVERT: ConvertFn<Self::Item> = AttributeKind::Link;
    const TEMPLATE: AttributeTemplate = template!(List: r#"name = "...", /*opt*/ kind = "dylib|static|...", /*opt*/ wasm_import_module = "...", /*opt*/ import_name_type = "decorated|noprefix|undecorated""#);

    fn extend<'c>(
        cx: &'c mut AcceptContext<'_, '_, S>,
        args: &'c ArgParser<'_>,
    ) -> impl IntoIterator<Item = Self::Item> + 'c {
        let mut result = None;
        let Some(items) = args.list() else {
            cx.expected_list(cx.attr_span);
            return result;
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

            match item.path().word().map(|ident| ident.name) {
                Some(sym::name) => {
                    if name.is_some() {
                        cx.duplicate_key(item.span(), sym::name);
                        continue;
                    }
                    let Some((link_name, link_name_span)) = item.args().name_value_str() else {
                        cx.expected_name_value(item.span(), Some(sym::name));
                        return result;
                    };
                    if link_name.is_empty() {
                        cx.emit_err(EmptyLinkName { span: link_name_span });
                    }
                    name = Some((link_name, link_name_span));
                }
                Some(sym::kind) => {
                    if kind.is_some() {
                        cx.duplicate_key(item.span(), sym::kind);
                        continue;
                    }
                    let Some((link_kind, link_kind_span)) = item.args().name_value_str() else {
                        cx.expected_name_value(item.span(), Some(sym::kind));
                        continue;
                    };

                    let link_kind = match link_kind.as_str() {
                        "static" => NativeLibKind::Static { bundle: None, whole_archive: None },
                        "dylib" => NativeLibKind::Dylib { as_needed: None },
                        "framework" => {
                            if !sess.target.is_like_darwin {
                                cx.emit_err(LinkFrameworkApple { span: link_kind_span });
                            }
                            NativeLibKind::Framework { as_needed: None }
                        }
                        "raw-dylib" => {
                            if sess.target.is_like_windows {
                                // raw-dylib is stable and working on Windows
                            } else if sess.target.binary_format == BinaryFormat::Elf
                                && features.raw_dylib_elf()
                            {
                                // raw-dylib is unstable on ELF, but the user opted in
                            } else if sess.target.binary_format == BinaryFormat::Elf
                                && sess.is_nightly_build()
                            {
                                feature_err(
                                    sess,
                                    sym::raw_dylib_elf,
                                    link_kind_span,
                                    fluent_generated::attr_parsing_raw_dylib_elf_unstable,
                                )
                                .emit();
                            } else {
                                cx.emit_err(RawDylibOnlyWindows { span: link_kind_span });
                            }

                            NativeLibKind::RawDylib
                        }
                        "link-arg" => {
                            if !features.link_arg_attribute() {
                                feature_err(
                                    sess,
                                    sym::link_arg_attribute,
                                    link_kind_span,
                                    fluent_generated::attr_parsing_link_arg_unstable,
                                )
                                .emit();
                            }
                            NativeLibKind::LinkArg
                        }
                        _kind => {
                            cx.expected_specific_argument_strings(
                                link_kind_span,
                                vec!["static", "dylib", "framework", "raw-dylib", "link-arg"],
                            );
                            continue;
                        }
                    };
                    kind = Some(link_kind);
                }
                Some(sym::modifiers) => {
                    if modifiers.is_some() {
                        cx.duplicate_key(item.span(), sym::modifiers);
                        continue;
                    }
                    let Some((link_modifiers, link_modifiers_span)) = item.args().name_value_str()
                    else {
                        cx.expected_name_value(item.span(), Some(sym::modifiers));
                        continue;
                    };
                    modifiers = Some((link_modifiers, link_modifiers_span));
                }
                Some(sym::cfg) => {
                    if cfg.is_some() {
                        cx.duplicate_key(item.span(), sym::cfg);
                        continue;
                    }
                    let Some(link_cfg) = item.args().list() else {
                        cx.expected_list(item.span());
                        continue;
                    };
                    let Some(link_cfg) = link_cfg.single() else {
                        cx.expected_single_argument(item.span());
                        continue;
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
                    cfg = parse_cfg_entry(cx, link_cfg);
                }
                Some(sym::wasm_import_module) => {
                    if wasm_import_module.is_some() {
                        cx.duplicate_key(item.span(), sym::wasm_import_module);
                        continue;
                    }
                    let Some((link_wasm_import_module, _)) = item.args().name_value_str() else {
                        cx.expected_name_value(item.span(), Some(sym::wasm_import_module));
                        continue;
                    };
                    wasm_import_module = Some((link_wasm_import_module, item.span()));
                }
                Some(sym::import_name_type) => {
                    if import_name_type.is_some() {
                        cx.duplicate_key(item.span(), sym::import_name_type);
                        continue;
                    }
                    let Some((link_import_name_type, _)) = item.args().name_value_str() else {
                        cx.expected_name_value(item.span(), Some(sym::import_name_type));
                        continue;
                    };
                    if cx.sess().target.arch != "x86" {
                        cx.emit_err(ImportNameTypeX86 { span: item.span() });
                        continue;
                    }

                    let link_import_name_type = match link_import_name_type.as_str() {
                        "decorated" => PeImportNameType::Decorated,
                        "noprefix" => PeImportNameType::NoPrefix,
                        "undecorated" => PeImportNameType::Undecorated,
                        _ => {
                            cx.expected_specific_argument_strings(
                                item.span(),
                                vec!["decorated", "noprefix", "undecorated"],
                            );
                            continue;
                        }
                    };
                    import_name_type = Some((link_import_name_type, item.span()));
                }
                _ => {
                    cx.expected_specific_argument_strings(
                        item.span(),
                        vec![
                            "name",
                            "kind",
                            "modifiers",
                            "cfg",
                            "wasm_import_module",
                            "import_name_type",
                        ],
                    );
                }
            }
        }

        // Do this outside the above loop so we don't depend on modifiers coming after kinds
        let mut verbatim = None;
        if let Some((modifiers, span)) = modifiers {
            for modifier in modifiers.as_str().split(',') {
                let (modifier, value) = match modifier.strip_prefix(&['+', '-']) {
                    Some(m) => (m, modifier.starts_with('+')),
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
                    ("bundle", Some(NativeLibKind::Static { bundle, .. })) => {
                        assign_modifier(bundle)
                    }
                    ("bundle", _) => {
                        cx.emit_err(BundleNeedsStatic { span });
                    }

                    ("verbatim", _) => assign_modifier(&mut verbatim),

                    ("whole-archive", Some(NativeLibKind::Static { whole_archive, .. })) => {
                        assign_modifier(whole_archive)
                    }
                    ("whole-archive", _) => {
                        cx.emit_err(WholeArchiveNeedsStatic { span });
                    }

                    ("as-needed", Some(NativeLibKind::Dylib { as_needed }))
                    | ("as-needed", Some(NativeLibKind::Framework { as_needed })) => {
                        report_unstable_modifier!(native_link_modifiers_as_needed);
                        assign_modifier(as_needed)
                    }
                    ("as-needed", _) => {
                        cx.emit_err(AsNeededCompatibility { span });
                    }

                    _ => {
                        cx.expected_specific_argument_strings(
                            span,
                            vec!["bundle", "verbatim", "whole-archive", "as-needed"],
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
            return result;
        };

        // Do this outside of the loop so that `import_name_type` can be specified before `kind`.
        if let Some((_, span)) = import_name_type {
            if kind != Some(NativeLibKind::RawDylib) {
                cx.emit_err(ImportNameTypeRaw { span });
            }
        }

        if let Some(NativeLibKind::RawDylib) = kind
            && name.as_str().contains('\0')
        {
            cx.emit_err(RawDylibNoNul { span: name_span });
        }

        result = Some(LinkEntry {
            span: cx.attr_span,
            kind: kind.unwrap_or(NativeLibKind::Unspecified),
            name,
            cfg,
            verbatim,
            import_name_type,
        });
        result
    }
}

pub(crate) struct LinkSectionParser;

impl<S: Stage> SingleAttributeParser<S> for LinkSectionParser {
    const PATH: &[Symbol] = &[sym::link_section];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepInnermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::WarnButFutureError;
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
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::ExportStable;
}

pub(crate) struct FfiConstParser;
impl<S: Stage> NoArgsAttributeParser<S> for FfiConstParser {
    const PATH: &[Symbol] = &[sym::ffi_const];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::FfiConst;
}

pub(crate) struct FfiPureParser;
impl<S: Stage> NoArgsAttributeParser<S> for FfiPureParser {
    const PATH: &[Symbol] = &[sym::ffi_pure];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Warn;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::FfiPure;
}

pub(crate) struct StdInternalSymbolParser;
impl<S: Stage> NoArgsAttributeParser<S> for StdInternalSymbolParser {
    const PATH: &[Symbol] = &[sym::rustc_std_internal_symbol];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const CREATE: fn(Span) -> AttributeKind = AttributeKind::StdInternalSymbol;
}

pub(crate) struct LinkOrdinalParser;

impl<S: Stage> SingleAttributeParser<S> for LinkOrdinalParser {
    const PATH: &[Symbol] = &[sym::link_ordinal];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepOutermost;
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const TEMPLATE: AttributeTemplate = template!(List: "ordinal");

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
