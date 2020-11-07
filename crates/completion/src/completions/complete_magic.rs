//! TODO kb move this into the complete_unqualified_path when starts to work properly

use assists::utils::{insert_use, mod_path_to_ast, ImportScope, MergeBehaviour};
use hir::Query;
use itertools::Itertools;
use syntax::{algo, AstNode};
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
            let mut builder = TextEdit::builder();

            let correct_qualifier = mod_path.segments.last()?.to_string();
            builder.replace(anchor.syntax().text_range(), correct_qualifier);

            // TODO kb: assists already have the merge behaviour setting, need to unite both
            let rewriter =
                insert_use(&import_scope, mod_path_to_ast(&mod_path), Some(MergeBehaviour::Full));
            let old_ast = rewriter.rewrite_root()?;
            algo::diff(&old_ast, &rewriter.rewrite(&old_ast)).into_text_edit(&mut builder);

            let completion_item: CompletionItem = CompletionItem::new(
                CompletionKind::Magic,
                ctx.source_range(),
                mod_path.to_string(),
            )
            .kind(CompletionItemKind::Struct)
            .text_edit(builder.finish())
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
    fn function_magic_completion() {
        check(
            r#"
//- /lib.rs crate:dep
pub mod io {
    pub fn stdin() {}
};

//- /main.rs crate:main deps:dep
fn main() {
    stdi<|>
}
"#,
            expect![[r#"
                st dep::io::stdin
            "#]],
        );

        check_edit(
            "dep::io::stdin",
            r#"
//- /lib.rs crate:dep
pub mod io {
    pub fn stdin() {}
};

//- /main.rs crate:main deps:dep
fn main() {
    stdi<|>
}
"#,
            r#"
use dep::io::stdin;

fn main() {
    stdin
}
"#,
        );
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
