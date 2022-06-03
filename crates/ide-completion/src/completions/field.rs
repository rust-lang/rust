//! Completion of field list position.

use crate::{
    context::{IdentContext, NameContext, NameKind, NameRefContext, PathCompletionCtx, PathKind},
    CompletionContext, CompletionItem, CompletionItemKind, Completions,
};

pub(crate) fn complete_field_list(acc: &mut Completions, ctx: &CompletionContext) {
    match &ctx.ident_ctx {
        IdentContext::Name(NameContext { kind: NameKind::RecordField, .. })
        | IdentContext::NameRef(NameRefContext {
            path_ctx:
                Some(PathCompletionCtx {
                    has_macro_bang: false,
                    is_absolute_path: false,
                    qualifier: None,
                    parent: None,
                    kind: PathKind::Type { in_tuple_struct: true },
                    has_type_args: false,
                    ..
                }),
            ..
        }) => {
            if ctx.qualifier_ctx.vis_node.is_none() {
                let mut add_keyword = |kw, snippet| add_keyword(acc, ctx, kw, snippet);
                add_keyword("pub(crate)", "pub(crate)");
                add_keyword("pub(super)", "pub(super)");
                add_keyword("pub", "pub");
            }
        }
        _ => return,
    }
}

pub(super) fn add_keyword(acc: &mut Completions, ctx: &CompletionContext, kw: &str, snippet: &str) {
    let mut item = CompletionItem::new(CompletionItemKind::Keyword, ctx.source_range(), kw);

    match ctx.config.snippet_cap {
        Some(cap) => {
            if snippet.ends_with('}') && ctx.incomplete_let {
                // complete block expression snippets with a trailing semicolon, if inside an incomplete let
                cov_mark::hit!(let_semi);
                item.insert_snippet(cap, format!("{};", snippet));
            } else {
                item.insert_snippet(cap, snippet);
            }
        }
        None => {
            item.insert_text(if snippet.contains('$') { kw } else { snippet });
        }
    };
    item.add_to(acc);
}
