use rustc_hir::attrs::AttributeKind::{LinkName, LinkOrdinal, LinkSection};
use rustc_hir::attrs::Linkage;

use super::prelude::*;
use super::util::parse_single_integer;
use crate::session_diagnostics::{LinkOrdinalOutOfRange, NullOnLinkSection};

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
