use rustc_ast::{DUMMY_NODE_ID, ItemKind, ast};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;

pub(crate) fn eii_macro_for(
    ecx: &mut ExtCtxt<'_>,
    _span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let Annotatable::Item(i) = &mut item else { panic!("expected item") };
    let ItemKind::MacroDef(d) = &mut i.kind else { panic!("expected macro def") };

    let Some(list) = meta_item.meta_item_list() else { panic!("expected list") };

    d.eii_macro_for = Some(list[0].meta_item().unwrap().path.clone());

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

    f.eii_impl.push((DUMMY_NODE_ID, meta_item.clone()));

    vec![item]
}
