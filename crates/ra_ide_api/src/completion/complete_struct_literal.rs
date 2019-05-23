use hir::AdtDef;

use crate::completion::{CompletionContext, Completions};

/// Complete fields in fields literals.
pub(super) fn complete_struct_literal(acc: &mut Completions, ctx: &CompletionContext) {
    let ty = match ctx.struct_lit_syntax.and_then(|it| ctx.analyzer.type_of(ctx.db, it.into())) {
        Some(it) => it,
        None => return,
    };
    let (adt, substs) = match ty.as_adt() {
        Some(res) => res,
        _ => return,
    };
    match adt {
        AdtDef::Struct(s) => {
            for field in s.fields(ctx.db) {
                acc.add_field(ctx, field, substs);
            }
        }

        // FIXME unions
        AdtDef::Union(_) => (),
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
        assert_debug_snapshot_matches!(completions, @r###"
       ⋮[
       ⋮    CompletionItem {
       ⋮        label: "the_field",
       ⋮        source_range: [83; 86),
       ⋮        delete: [83; 86),
       ⋮        insert: "the_field",
       ⋮        kind: Field,
       ⋮        detail: "u32",
       ⋮    },
       ⋮]
        "###);
    }
}
