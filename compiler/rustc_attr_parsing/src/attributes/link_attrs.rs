use rustc_attr_data_structures::AttributeKind;
use rustc_attr_data_structures::AttributeKind::{LinkName, LinkSection};
use rustc_feature::{AttributeTemplate, template};
use rustc_span::{Span, Symbol, sym};

use crate::attributes::{
    AttributeOrder, NoArgsAttributeParser, OnDuplicate, SingleAttributeParser,
};
use crate::context::{AcceptContext, Stage};
use crate::parser::ArgParser;
use crate::session_diagnostics::NullOnLinkSection;

pub(crate) struct LinkNameParser;

impl<S: Stage> SingleAttributeParser<S> for LinkNameParser {
    const PATH: &[Symbol] = &[sym::link_name];
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
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
    const ATTRIBUTE_ORDER: AttributeOrder = AttributeOrder::KeepFirst;
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
