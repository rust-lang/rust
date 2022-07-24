//! Completion of field list position.

use crate::{
    context::{PathCompletionCtx, Qualified},
    CompletionContext, Completions,
};

pub(crate) fn complete_field_list_tuple_variant(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    path_ctx: &PathCompletionCtx,
) {
    if ctx.qualifier_ctx.vis_node.is_some() {
        return;
    }
    match path_ctx {
        PathCompletionCtx {
            has_macro_bang: false,
            qualified: Qualified::No,
            parent: None,
            has_type_args: false,
            ..
        } => {
            let mut add_keyword = |kw, snippet| acc.add_keyword_snippet(ctx, kw, snippet);
            add_keyword("pub(crate)", "pub(crate)");
            add_keyword("pub(super)", "pub(super)");
            add_keyword("pub", "pub");
        }
        _ => (),
    }
}

pub(crate) fn complete_field_list_record_variant(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
) {
    if ctx.qualifier_ctx.vis_node.is_none() {
        let mut add_keyword = |kw, snippet| acc.add_keyword_snippet(ctx, kw, snippet);
        add_keyword("pub(crate)", "pub(crate)");
        add_keyword("pub(super)", "pub(super)");
        add_keyword("pub", "pub");
    }
}
