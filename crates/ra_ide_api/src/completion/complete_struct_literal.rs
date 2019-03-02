use hir::{Ty, AdtDef};

use crate::completion::{CompletionContext, Completions};

/// Complete fields in fields literals.
pub(super) fn complete_struct_literal(acc: &mut Completions, ctx: &CompletionContext) {
    let (function, struct_lit) = match (&ctx.function, ctx.struct_lit_syntax) {
        (Some(function), Some(struct_lit)) => (function, struct_lit),
        _ => return,
    };
    let infer_result = function.infer(ctx.db);
    let source_map = function.body_source_map(ctx.db);
    let expr = match source_map.node_expr(struct_lit.into()) {
        Some(expr) => expr,
        None => return,
    };
    let ty = infer_result[expr].clone();
    let (adt, substs) = match ty {
        Ty::Adt { def_id, ref substs, .. } => (def_id, substs),
        _ => return,
    };
    match adt {
        AdtDef::Struct(s) => {
            for field in s.fields(ctx.db) {
                acc.add_field(ctx, field, substs);
            }
        }

        // TODO unions
        AdtDef::Enum(_) => (),
    };
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot_matches;
    use crate::completion::{CompletionItem, CompletionKind, do_completion};

    fn complete(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
    }

    #[test]
    fn test_struct_literal_field() {
        let completions = complete(
            r"
            struct A { the_field: u32 }
            fn foo() {
               A { the<|> }
            }
            ",
        );
        assert_debug_snapshot_matches!(completions, @r###"[
    CompletionItem {
        label: "the_field",
        source_range: [83; 86),
        delete: [83; 86),
        insert: "the_field",
        kind: Field,
        detail: "u32"
    }
]"###);
    }
}
