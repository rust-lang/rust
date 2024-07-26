use rustc_ast::{DUMMY_NODE_ID, ast};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;

pub(crate) fn expand(
    ecx: &mut ExtCtxt<'_>,
    _expand_span: Span,
    meta_item: &ast::MetaItem,
    mut item: Annotatable,
) -> Vec<Annotatable> {
    let define_opaques = match &mut item {
        Annotatable::Item(p) => match p.define_opaques() {
            Some(g) => g,
            None => {
                ecx.dcx().span_err(meta_item.span, "only functions can define opaque types");
                return vec![item];
            }
        },
        Annotatable::AssocItem(i, _assoc_ctxt) => match &mut i.kind {
            ast::AssocItemKind::Fn(func) => &mut func.generics.define_opaques,
            ast::AssocItemKind::Const(c) => &mut c.generics.define_opaques,
            ast::AssocItemKind::Type(_)
            | ast::AssocItemKind::MacCall(_)
            | ast::AssocItemKind::Delegation(_)
            | ast::AssocItemKind::DelegationMac(_) => {
                ecx.dcx()
                    .span_err(meta_item.span, "only associated functions can define opaque types");
                return vec![item];
            }
        },
        Annotatable::Stmt(s) => match &mut s.kind {
            ast::StmtKind::Item(p) => match p.define_opaques() {
                Some(g) => g,
                None => {
                    ecx.dcx().span_err(meta_item.span, "only functions can define opaque types");
                    return vec![item];
                }
            },
            ast::StmtKind::Let(_)
            | ast::StmtKind::Expr(_)
            | ast::StmtKind::Semi(_)
            | ast::StmtKind::Empty
            | ast::StmtKind::MacCall(_) => {
                ecx.dcx().span_err(meta_item.span, "only items can define opaque types");
                return vec![item];
            }
        },
        Annotatable::Expr(e) => match &mut e.kind {
            ast::ExprKind::Closure(closure) => &mut closure.define_opaques,
            _ => {
                ecx.dcx()
                    .span_err(meta_item.span, "only closure expressions can define opaque types");
                return vec![item];
            }
        },
        Annotatable::ForeignItem(_)
        | Annotatable::Arm(_)
        | Annotatable::ExprField(_)
        | Annotatable::PatField(_)
        | Annotatable::GenericParam(_)
        | Annotatable::Param(_)
        | Annotatable::FieldDef(_)
        | Annotatable::Variant(_)
        | Annotatable::Crate(_) => {
            ecx.dcx().span_err(meta_item.span, "only items can define opaque types");
            return vec![item];
        }
    };

    let Some(list) = meta_item.meta_item_list() else {
        ecx.dcx().span_err(meta_item.span, "expected list of type aliases");
        return vec![item];
    };

    *define_opaques = Some(
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

    vec![item]
}
