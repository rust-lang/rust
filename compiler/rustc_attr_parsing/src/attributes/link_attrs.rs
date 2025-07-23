use rustc_attr_data_structures::AttributeKind;
use rustc_attr_data_structures::AttributeKind::{LinkName, LinkOrdinal, LinkSection};
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{
    AttributeOrder, NoArgsAttributeParser, OnDuplicate, SingleAttributeParser,
};
use crate::context::{AcceptContext, Stage, parse_single_integer};
use crate::parser::ArgParser;
use crate::session_diagnostics::{LinkOrdinalOutOfRange, NullOnLinkSection};

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
