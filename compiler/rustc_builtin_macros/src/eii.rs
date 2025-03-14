use rustc_ast::{DUMMY_NODE_ID, EIIImpl, EiiMacroFor, ItemKind, ast};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::{Span, kw};

pub(crate) fn eii_macro_for(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let Annotatable::Item(i) = &mut item else { panic!("expected item") };
    let ItemKind::MacroDef(_, d) = &mut i.kind else { panic!("expected macro def") };

    let Some(list) = meta_item.meta_item_list() else { panic!("expected list") };

    let Some(extern_item_path) = list.get(0).and_then(|i| i.meta_item()).map(|i| i.path.clone())
    else {
        panic!("expected a path to an `extern` item");
    };

    let impl_unsafe = if let Some(i) = list.get(1) {
        if i.lit().and_then(|i| i.kind.str()).is_some_and(|i| i == kw::Unsafe) {
            true
        } else {
            panic!("expected the string `\"unsafe\"` here or no other arguments");
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
    let Annotatable::Item(i) = &mut item else { panic!("expected item") };

    let ItemKind::Fn(f) = &mut i.kind else { panic!("expected function") };

    assert!(meta_item.is_word());

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
