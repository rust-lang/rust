//! TODO kb move this into the complete_unqualified_path when starts to work properly

use assists::utils::{insert_use, mod_path_to_ast, ImportScope};
use either::Either;
use hir::ScopeDef;
use ide_db::imports_locator;
use syntax::{algo, AstNode};

use crate::{
    context::CompletionContext,
    render::{render_resolution, RenderContext},
};

use super::Completions;

// TODO kb add a setting toggle for this feature?
pub(crate) fn complete_magic(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if !(ctx.is_trivial_path || ctx.is_pat_binding_or_const) {
        return None;
    }
    let current_module = ctx.scope.module()?;
    let anchor = ctx.name_ref_syntax.as_ref()?;
    let import_scope = ImportScope::find_insert_use_container(anchor.syntax(), &ctx.sema)?;

    let potential_import_name = ctx.token.to_string();

    let possible_imports =
        imports_locator::find_similar_imports(&ctx.sema, ctx.krate?, &potential_import_name)
            .filter_map(|import_candidate| {
                Some(match import_candidate {
                    Either::Left(module_def) => (
                        current_module.find_use_path(ctx.db, module_def)?,
                        ScopeDef::ModuleDef(module_def),
                    ),
                    Either::Right(macro_def) => (
                        current_module.find_use_path(ctx.db, macro_def)?,
                        ScopeDef::MacroDef(macro_def),
                    ),
                })
            })
            .filter_map(|(mod_path, definition)| {
                let mut resolution_with_missing_import = render_resolution(
                    RenderContext::new(ctx),
                    mod_path.segments.last()?.to_string(),
                    &definition,
                )?;

                let mut text_edits =
                    resolution_with_missing_import.text_edit().to_owned().into_builder();

                let rewriter =
                    insert_use(&import_scope, mod_path_to_ast(&mod_path), ctx.config.merge);
                let old_ast = rewriter.rewrite_root()?;
                algo::diff(&old_ast, &rewriter.rewrite(&old_ast)).into_text_edit(&mut text_edits);

                resolution_with_missing_import.update_text_edit(text_edits.finish());

                Some(resolution_with_missing_import)
            });

    acc.add_all(possible_imports);
    Some(())
}

#[cfg(test)]
mod tests {
    use crate::test_utils::check_edit;

    #[test]
    fn function_magic_completion() {
        check_edit(
            "stdin",
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
    stdin()$0
}
"#,
        );
    }

    #[test]
    fn macro_magic_completion() {
        check_edit(
            "macro_with_curlies!",
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
    macro_with_curlies! {$0}
}
"#,
        );
    }

    #[test]
    fn case_insensitive_magic_completion_works() {
        check_edit(
            "ThirdStruct",
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
