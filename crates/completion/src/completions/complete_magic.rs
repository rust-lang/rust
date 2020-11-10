//! TODO kb move this into the complete_unqualified_path when starts to work properly

use assists::utils::{insert_use, mod_path_to_ast, ImportScope};
use either::Either;
use hir::{db::HirDatabase, MacroDef, ModuleDef, Query};
use itertools::Itertools;
use syntax::{algo, AstNode};
use text_edit::TextEdit;

use crate::{context::CompletionContext, item::CompletionKind, CompletionItem, CompletionItemKind};

use super::Completions;

// TODO kb when typing, completes partial results, need to rerun manually to see the proper ones
pub(crate) fn complete_magic(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if !(ctx.is_trivial_path || ctx.is_pat_binding_or_const) {
        return None;
    }
    let current_module = ctx.scope.module()?;
    let anchor = ctx.name_ref_syntax.as_ref()?;
    let import_scope = ImportScope::find_insert_use_container(anchor.syntax(), &ctx.sema)?;

    // TODO kb consider heuristics, such as "don't show `hash_map` import if `HashMap` is the import for completion"
    // also apply completion ordering
    let potential_import_name = ctx.token.to_string();

    let possible_imports = ctx
        .krate?
        // TODO kb use imports_locator instead?
        .query_external_importables(ctx.db, Query::new(&potential_import_name).limit(40))
        .unique()
        .filter_map(|import_candidate| {
            let use_path = match import_candidate {
                Either::Left(module_def) => current_module.find_use_path(ctx.db, module_def),
                Either::Right(macro_def) => current_module.find_use_path(ctx.db, macro_def),
            }?;
            // TODO kb need to omit braces when there are some already.
            // maybe remove braces completely?
            Some((use_path, additional_completion(ctx.db, import_candidate)))
        })
        .filter_map(|(mod_path, additional_completion)| {
            let mut builder = TextEdit::builder();

            let correct_qualifier = format!(
                "{}{}",
                mod_path.segments.last()?,
                additional_completion.unwrap_or_default()
            );
            builder.replace(anchor.syntax().text_range(), correct_qualifier);

            let rewriter = insert_use(&import_scope, mod_path_to_ast(&mod_path), ctx.config.merge);
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

fn additional_completion(
    db: &dyn HirDatabase,
    import_candidate: Either<ModuleDef, MacroDef>,
) -> Option<String> {
    match import_candidate {
        Either::Left(ModuleDef::Function(_)) => Some("()".to_string()),
        Either::Right(macro_def) => {
            let (left_brace, right_brace) =
                crate::render::macro_::guess_macro_braces(db, macro_def);
            Some(format!("!{}{}", left_brace, right_brace))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::check_edit;

    #[test]
    fn function_magic_completion() {
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
    stdin()
}
"#,
        );
    }

    #[test]
    fn macro_magic_completion() {
        check_edit(
            "dep::macro_with_curlies",
            r#"
//- /lib.rs crate:dep
/// Please call me as macro_with_curlies! {}
#[macro_export]
macro_rules! macro_with_curlies {
    () => {}
}

//- /main.rs crate:main deps:dep
fn main() {
    curli<|>
}
"#,
            r#"
use dep::macro_with_curlies;

fn main() {
    macro_with_curlies! {}
}
"#,
        );
    }

    #[test]
    fn case_insensitive_magic_completion_works() {
        check_edit(
            "dep::some_module::ThirdStruct",
            r#"
//- /lib.rs crate:dep
pub struct FirstStruct;
pub mod some_module {
    pub struct SecondStruct;
    pub struct ThirdStruct;
}

//- /main.rs crate:main deps:dep
use dep::{FirstStruct, some_module::SecondStruct};

fn main() {
    this<|>
}
"#,
            r#"
use dep::{FirstStruct, some_module::{SecondStruct, ThirdStruct}};

fn main() {
    ThirdStruct
}
"#,
        );
    }
}
