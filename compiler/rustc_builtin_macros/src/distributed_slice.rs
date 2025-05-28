use rustc_ast::{DistributedSlice, ItemKind, ast};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::Span;

pub(crate) fn distributed_slice(
    _ecx: &mut ExtCtxt<'_>,
    span: Span,
    _meta_item: &ast::MetaItem,
    mut orig_item: Annotatable,
) -> Vec<Annotatable> {
    // TODO: FIXME(gr)
    // FIXME(gr): check item

    let Annotatable::Item(item) = &mut orig_item else {
        panic!("expected `#[distributed_slice]` on an item")
    };

    match &mut item.kind {
        ItemKind::Static(static_item) => {
            static_item.distributed_slice = DistributedSlice::Declaration(span);
        }
        ItemKind::Const(const_item) => {
            const_item.distributed_slice = DistributedSlice::Declaration(span);
        }
        other => {
            panic!("expected `#[distributed_slice]` on a const or static item, not {other:?}");
        }
    }

    vec![orig_item]
}
