use rustc_ast::{DUMMY_NODE_ID, EIIImpl, EiiMacroFor, ItemKind, ast};
use rustc_ast_pretty::pprust::path_to_string;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Span, kw};

use crate::errors::{
    EIIMacroExpectedFunction, EIIMacroExpectedMaxOneArgument, EIIMacroForExpectedList,
    EIIMacroForExpectedMacro, EIIMacroForExpectedUnsafe,
};

pub(crate) fn eii_macro_for(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let Annotatable::Item(i) = &mut item else {
        ecx.dcx().emit_err(EIIMacroForExpectedMacro { span });
        return vec![item];
    };
    let ItemKind::MacroDef(_, d) = &mut i.kind else {
        ecx.dcx().emit_err(EIIMacroForExpectedMacro { span });
        return vec![item];
    };

    let Some(list) = meta_item.meta_item_list() else {
        ecx.dcx().emit_err(EIIMacroForExpectedList { span: meta_item.span });
        return vec![item];
    };

    if list.len() > 2 {
        ecx.dcx().emit_err(EIIMacroForExpectedList { span: meta_item.span });
        return vec![item];
    }

    let Some(extern_item_path) = list.get(0).and_then(|i| i.meta_item()).map(|i| i.path.clone())
    else {
        ecx.dcx().emit_err(EIIMacroForExpectedList { span: meta_item.span });
        return vec![item];
    };

    let impl_unsafe = if let Some(i) = list.get(1) {
        if i.lit().and_then(|i| i.kind.str()).is_some_and(|i| i == kw::Unsafe) {
            true
        } else {
            ecx.dcx().emit_err(EIIMacroForExpectedUnsafe { span: i.span() });
            return vec![item];
        }
    } else {
        false
    };

    d.eii_macro_for = Some(EiiMacroFor { extern_item_path, impl_unsafe });

    // Return the original item and the new methods.
    vec![item]
}

pub(crate) fn eii_macro(
    ecx: &mut ExtCtxt<'_>,
    span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let Annotatable::Item(i) = &mut item else {
        ecx.dcx()
            .emit_err(EIIMacroExpectedFunction { span, name: path_to_string(&meta_item.path) });
        return vec![item];
    };

    let ItemKind::Fn(f) = &mut i.kind else {
        ecx.dcx()
            .emit_err(EIIMacroExpectedFunction { span, name: path_to_string(&meta_item.path) });
        return vec![item];
    };

    let is_default = if meta_item.is_word() {
        false
    } else if let Some([first]) = meta_item.meta_item_list()
        && let Some(m) = first.meta_item()
        && m.path.segments.len() == 1
    {
        m.path.segments[0].ident.name == kw::Default
    } else {
        ecx.dcx().emit_err(EIIMacroExpectedMaxOneArgument {
            span: meta_item.span,
            name: path_to_string(&meta_item.path),
        });
        return vec![item];
    };

    f.eii_impl.push(EIIImpl {
        node_id: DUMMY_NODE_ID,
        eii_macro_path: meta_item.path.clone(),
        impl_safety: meta_item.unsafety,
        span,
        inner_span: meta_item.span,
        is_default: false,
    });

    vec![item]
}
