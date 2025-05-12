use rustc_ast::{DUMMY_NODE_ID, ast};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;

pub(crate) fn expand(
    ecx: &mut ExtCtxt<'_>,
    _expand_span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let define_opaque = match &mut item {
        Annotatable::Item(p) => match &mut p.kind {
            ast::ItemKind::Fn(f) => Some(&mut f.define_opaque),
            ast::ItemKind::Const(ct) => Some(&mut ct.define_opaque),
            ast::ItemKind::Static(si) => Some(&mut si.define_opaque),
            _ => None,
        },
        Annotatable::AssocItem(i, _assoc_ctxt) => match &mut i.kind {
            ast::AssocItemKind::Fn(func) => Some(&mut func.define_opaque),
            ast::AssocItemKind::Const(ct) => Some(&mut ct.define_opaque),
            _ => None,
        },
        Annotatable::Stmt(s) => match &mut s.kind {
            ast::StmtKind::Item(p) => match &mut p.kind {
                ast::ItemKind::Fn(f) => Some(&mut f.define_opaque),
                ast::ItemKind::Const(ct) => Some(&mut ct.define_opaque),
                ast::ItemKind::Static(si) => Some(&mut si.define_opaque),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    };

    let Some(list) = meta_item.meta_item_list() else {
        ecx.dcx().span_err(meta_item.span, "expected list of type aliases");
        return vec![item];
    };

    if let Some(define_opaque) = define_opaque {
        *define_opaque = Some(
            list.iter()
                .filter_map(|entry| match entry {
                    ast::MetaItemInner::MetaItem(meta_item) if meta_item.is_word() => {
                        Some((DUMMY_NODE_ID, meta_item.path.clone()))
                    }
                    _ => {
                        ecx.dcx().span_err(entry.span(), "expected path to type alias");
                        None
                    }
                })
                .collect(),
        );
    } else {
        ecx.dcx().span_err(
            meta_item.span,
            "only functions, statics, and consts can define opaque types",
        );
    }

    vec![item]
}
