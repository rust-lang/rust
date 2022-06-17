//! Completion of field list position.

use crate::{
    context::{NameContext, NameKind, PathCompletionCtx, PathKind, Qualified, TypeLocation},
    CompletionContext, Completions,
};

pub(crate) fn complete_field_list_tuple_variant(
    acc: &mut Completions,
    ctx: &CompletionContext,
    path_ctx: &PathCompletionCtx,
) {
    match path_ctx {
        PathCompletionCtx {
            has_macro_bang: false,
            qualified: Qualified::No,
            parent: None,
            kind: PathKind::Type { location: TypeLocation::TupleField },
            has_type_args: false,
            ..
        } => {
            if ctx.qualifier_ctx.vis_node.is_none() {
                let mut add_keyword = |kw, snippet| acc.add_keyword_snippet(ctx, kw, snippet);
                add_keyword("pub(crate)", "pub(crate)");
                add_keyword("pub(super)", "pub(super)");
                add_keyword("pub", "pub");
            }
        }
        _ => (),
    }
}

pub(crate) fn complete_field_list_record_variant(
    acc: &mut Completions,
    ctx: &CompletionContext,
    name_ctx: &NameContext,
) {
    if let NameContext { kind: NameKind::RecordField, .. } = name_ctx {
        if ctx.qualifier_ctx.vis_node.is_none() {
            let mut add_keyword = |kw, snippet| acc.add_keyword_snippet(ctx, kw, snippet);
            add_keyword("pub(crate)", "pub(crate)");
            add_keyword("pub(super)", "pub(super)");
            add_keyword("pub", "pub");
        }
    }
}
