//! Feature: completion with imports-on-the-fly
//!
//! When completing names in the current scope, proposes additional imports from other modules or crates,
//! if they can be qualified in the scope and their name contains all symbols from the completion input
//! (case-insensitive, in any order or places).
//!
//! ```
//! fn main() {
//!     pda$0
//! }
//! # pub mod std { pub mod marker { pub struct PhantomData { } } }
//! ```
//! ->
//! ```
//! use std::marker::PhantomData;
//!
//! fn main() {
//!     PhantomData
//! }
//! # pub mod std { pub mod marker { pub struct PhantomData { } } }
//! ```
//!
//! .Fuzzy search details
//!
//! To avoid an excessive amount of the results returned, completion input is checked for inclusion in the names only
//! (i.e. in `HashMap` in the `std::collections::HashMap` path).
//! For the same reasons, avoids searching for any imports for inputs with their length less that 2 symbols.
//!
//! .Import configuration
//!
//! It is possible to configure how use-trees are merged with the `importMergeBehavior` setting.
//! Mimics the corresponding behavior of the `Auto Import` feature.
//!
//! .LSP and performance implications
//!
//! The feature is enabled only if the LSP client supports LSP protocol version 3.16+ and reports the `additionalTextEdits`
//! (case sensitive) resolve client capability in its client capabilities.
//! This way the server is able to defer the costly computations, doing them for a selected completion item only.
//! For clients with no such support, all edits have to be calculated on the completion request, including the fuzzy search completion ones,
//! which might be slow ergo the feature is automatically disabled.
//!
//! .Feature toggle
//!
//! The feature can be forcefully turned off in the settings with the `rust-analyzer.completion.enableAutoimportCompletions` flag.
//! Note that having this flag set to `true` does not guarantee that the feature is enabled: your client needs to have the corredponding
//! capability enabled.

use either::Either;
use hir::{ModPath, ScopeDef};
use ide_db::{helpers::insert_use::ImportScope, imports_locator};
use syntax::AstNode;
use test_utils::mark;

use crate::{
    context::CompletionContext,
    render::{render_resolution_with_import, RenderContext},
    ImportEdit,
};

use super::Completions;

pub(crate) fn import_on_the_fly(acc: &mut Completions, ctx: &CompletionContext) -> Option<()> {
    if !ctx.config.enable_autoimport_completions {
        return None;
    }
    if ctx.attribute_under_caret.is_some() || ctx.mod_declaration_under_caret.is_some() {
        return None;
    }
    let potential_import_name = ctx.token.to_string();
    if potential_import_name.len() < 2 {
        return None;
    }
    let _p = profile::span("import_on_the_fly").detail(|| potential_import_name.to_string());

    let current_module = ctx.scope.module()?;
    let anchor = ctx.name_ref_syntax.as_ref()?;
    let import_scope = ImportScope::find_insert_use_container(anchor.syntax(), &ctx.sema)?;

    let user_input_lowercased = potential_import_name.to_lowercase();
    let mut all_mod_paths = imports_locator::find_similar_imports(
        &ctx.sema,
        ctx.krate?,
        Some(40),
        potential_import_name,
        true,
        true,
    )
    .filter_map(|import_candidate| {
        Some(match import_candidate {
            Either::Left(module_def) => {
                (current_module.find_use_path(ctx.db, module_def)?, ScopeDef::ModuleDef(module_def))
            }
            Either::Right(macro_def) => {
                (current_module.find_use_path(ctx.db, macro_def)?, ScopeDef::MacroDef(macro_def))
            }
        })
    })
    .filter(|(mod_path, _)| mod_path.len() > 1)
    .collect::<Vec<_>>();

    all_mod_paths.sort_by_cached_key(|(mod_path, _)| {
        compute_fuzzy_completion_order_key(mod_path, &user_input_lowercased)
    });

    acc.add_all(all_mod_paths.into_iter().filter_map(|(import_path, definition)| {
        render_resolution_with_import(
            RenderContext::new(ctx),
            ImportEdit { import_path, import_scope: import_scope.clone() },
            &definition,
        )
    }));
    Some(())
}

fn compute_fuzzy_completion_order_key(
    proposed_mod_path: &ModPath,
    user_input_lowercased: &str,
) -> usize {
    mark::hit!(certain_fuzzy_order_test);
    let proposed_import_name = match proposed_mod_path.segments.last() {
        Some(name) => name.to_string().to_lowercase(),
        None => return usize::MAX,
    };
    match proposed_import_name.match_indices(user_input_lowercased).next() {
        Some((first_matching_index, _)) => first_matching_index,
        None => usize::MAX,
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use test_utils::mark;

    use crate::{
        item::CompletionKind,
        test_utils::{check_edit, completion_list},
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Magic);
        expect.assert_eq(&actual);
    }

    #[test]
    fn function_fuzzy_completion() {
        check_edit(
            "stdin",
            r#"
//- /lib.rs crate:dep
pub mod io {
    pub fn stdin() {}
};

//- /main.rs crate:main deps:dep
fn main() {
    stdi$0
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
    fn macro_fuzzy_completion() {
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
    curli$0
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
    fn struct_fuzzy_completion() {
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
    this$0
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

    #[test]
    fn fuzzy_completions_come_in_specific_order() {
        mark::check!(certain_fuzzy_order_test);
        check(
            r#"
//- /lib.rs crate:dep
pub struct FirstStruct;
pub mod some_module {
    // already imported, omitted
    pub struct SecondStruct;
    // does not contain all letters from the query, omitted
    pub struct UnrelatedOne;
    // contains all letters from the query, but not in sequence, displayed last
    pub struct ThiiiiiirdStruct;
    // contains all letters from the query, but not in the beginning, displayed second
    pub struct AfterThirdStruct;
    // contains all letters from the query in the begginning, displayed first
    pub struct ThirdStruct;
}

//- /main.rs crate:main deps:dep
use dep::{FirstStruct, some_module::SecondStruct};

fn main() {
    hir$0
}
"#,
            expect![[r#"
                st dep::some_module::ThirdStruct
                st dep::some_module::AfterThirdStruct
                st dep::some_module::ThiiiiiirdStruct
            "#]],
        );
    }

    #[test]
    fn does_not_propose_names_in_scope() {
        check(
            r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
use dep::test_mod::TestStruct;
fn main() {
    TestSt$0
}
"#,
            expect![[r#""#]],
        );
    }
}
