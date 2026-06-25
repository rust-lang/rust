use rustc_ast::ast;
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;

pub(crate) fn expand(
    ecx: &mut ExtCtxt<'_>,
    _expand_span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let constness = match &mut item {
        Annotatable::Item(p) => match &mut p.kind {
            ast::ItemKind::Fn(f) => Some(&mut f.sig.header.constness),
            _ => None,
        },
        Annotatable::AssocItem(i, _assoc_ctxt) => match &mut i.kind {
            ast::AssocItemKind::Fn(func) => Some(&mut func.sig.header.constness),
            _ => None,
        },
        Annotatable::Stmt(s) => match &mut s.kind {
            ast::StmtKind::Item(p) => match &mut p.kind {
                ast::ItemKind::Fn(f) => Some(&mut f.sig.header.constness),
                _ => None,
            },
            _ => None,
        },
        _ => None,
    };

    let ast::MetaItemKind::Word = meta_item.kind else {
        ecx.dcx().span_err(meta_item.span, "comptime does not take any arguments");
        return vec![item];
    };

    if let Some(constness) = constness {
        if let ast::Const::Yes(span) = *constness {
            ecx.dcx().span_err(span, "a function cannot be both `comptime` and `const`");
        }
        *constness = ast::Const::Always(meta_item.span);
    } else {
        ecx.dcx().span_err(meta_item.span, "only functions and methods may be comptime");
    }

    vec![item]
}
