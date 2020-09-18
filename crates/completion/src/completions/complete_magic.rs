//! TODO kb move this into the complete_unqualified_path when starts to work properly

use assists::utils::{insert_use, mod_path_to_ast, ImportScope, MergeBehaviour};
use hir::Query;
use itertools::Itertools;
use syntax::AstNode;
use text_edit::TextEdit;

use crate::{context::CompletionContext, item::CompletionKind, CompletionItem, CompletionItemKind};

use super::Completions;

pub(crate) fn complete_magic(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if !(ctx.is_trivial_path || ctx.is_pat_binding_or_const) {
        return None;
    }
    let current_module = ctx.scope.module()?;
    let anchor = ctx.name_ref_syntax.as_ref()?;
    let import_scope = ImportScope::find_insert_use_container(anchor.syntax(), &ctx.sema)?;
    // TODO kb now this is the whole file, which is not disjoint with any other change in the same file, fix it
    // otherwise it's impossible to correctly add the use statement and also change the completed text into something more meaningful
    let import_syntax = import_scope.as_syntax_node();

    // TODO kb consider heuristics, such as "don't show `hash_map` import if `HashMap` is the import for completion"
    // TODO kb module functions are not completed, consider `std::io::stdin` one
    let potential_import_name = ctx.token.to_string();

    let possible_imports = ctx
        .krate?
        // TODO kb use imports_locator instead?
        .query_external_importables(ctx.db, Query::new(&potential_import_name).limit(40))
        .unique()
        .filter_map(|import_candidate| match import_candidate {
            either::Either::Left(module_def) => current_module.find_use_path(ctx.db, module_def),
            either::Either::Right(macro_def) => current_module.find_use_path(ctx.db, macro_def),
        })
        .filter_map(|mod_path| {
            let correct_qualifier = mod_path.segments.last()?.to_string();
            let rewriter =
                insert_use(&import_scope, mod_path_to_ast(&mod_path), Some(MergeBehaviour::Full));
            let rewritten_node = rewriter.rewrite(import_syntax);
            let insert_use_edit =
                TextEdit::replace(import_syntax.text_range(), rewritten_node.to_string());
            let mut completion_edit =
                TextEdit::replace(anchor.syntax().text_range(), correct_qualifier);
            completion_edit.union(insert_use_edit).expect("TODO kb");

            let completion_item: CompletionItem = CompletionItem::new(
                CompletionKind::Magic,
                ctx.source_range(),
                mod_path.to_string(),
            )
            .kind(CompletionItemKind::Struct)
            .text_edit(completion_edit)
            .into();
            Some(completion_item)
        });
    acc.add_all(possible_imports);

    Some(())
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{
        item::CompletionKind,
        test_utils::{check_edit, completion_list},
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Magic);
        expect.assert_eq(&actual)
    }

    #[test]
    fn case_insensitive_magic_completion_works() {
        check(
            r#"
//- /lib.rs crate:dep
pub struct TestStruct;

//- /main.rs crate:main deps:dep
fn main() {
    teru<|>
}
"#,
            expect![[r#"
                st dep::TestStruct
            "#]],
        );

        check_edit(
            "dep::TestStruct",
            r#"
//- /lib.rs crate:dep
pub struct TestStruct;

//- /main.rs crate:main deps:dep
fn main() {
    teru<|>
}
"#,
            r#"
use dep::TestStruct;

fn main() {
    TestStruct
}
"#,
        );
    }
}
