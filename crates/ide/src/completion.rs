mod completion_config;
mod completion_item;
mod completion_context;
mod presentation;
mod patterns;
mod generated_features;
#[cfg(test)]
mod test_utils;

mod complete_attribute;
mod complete_dot;
mod complete_record;
mod complete_pattern;
mod complete_fn_param;
mod complete_keyword;
mod complete_snippet;
mod complete_qualified_path;
mod complete_unqualified_path;
mod complete_postfix;
mod complete_macro_in_item_position;
mod complete_trait_impl;
mod complete_mod;

use ide_db::RootDatabase;

use crate::{
    completion::{
        completion_context::CompletionContext,
        completion_item::{CompletionKind, Completions},
    },
    FilePosition,
};

pub use crate::completion::{
    completion_config::CompletionConfig,
    completion_item::{CompletionItem, CompletionItemKind, CompletionScore, InsertTextFormat},
};

//FIXME: split the following feature into fine-grained features.

// Feature: Magic Completions
//
// In addition to usual reference completion, rust-analyzer provides some ✨magic✨
// completions as well:
//
// Keywords like `if`, `else` `while`, `loop` are completed with braces, and cursor
// is placed at the appropriate position. Even though `if` is easy to type, you
// still want to complete it, to get ` { }` for free! `return` is inserted with a
// space or `;` depending on the return type of the function.
//
// When completing a function call, `()` are automatically inserted. If a function
// takes arguments, the cursor is positioned inside the parenthesis.
//
// There are postfix completions, which can be triggered by typing something like
// `foo().if`. The word after `.` determines postfix completion. Possible variants are:
//
// - `expr.if` -> `if expr {}` or `if let ... {}` for `Option` or `Result`
// - `expr.match` -> `match expr {}`
// - `expr.while` -> `while expr {}` or `while let ... {}` for `Option` or `Result`
// - `expr.ref` -> `&expr`
// - `expr.refm` -> `&mut expr`
// - `expr.not` -> `!expr`
// - `expr.dbg` -> `dbg!(expr)`
//
// There also snippet completions:
//
// .Expressions
// - `pd` -> `eprintln!(" = {:?}", );`
// - `ppd` -> `eprintln!(" = {:#?}", );`
//
// .Items
// - `tfn` -> `#[test] fn feature(){}`
// - `tmod` ->
// ```rust
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_name() {}
// }
// ```

/// Main entry point for completion. We run completion as a two-phase process.
///
/// First, we look at the position and collect a so-called `CompletionContext.
/// This is a somewhat messy process, because, during completion, syntax tree is
/// incomplete and can look really weird.
///
/// Once the context is collected, we run a series of completion routines which
/// look at the context and produce completion items. One subtlety about this
/// phase is that completion engine should not filter by the substring which is
/// already present, it should give all possible variants for the identifier at
/// the caret. In other words, for
///
/// ```no_run
/// fn f() {
///     let foo = 92;
///     let _ = bar<|>
/// }
/// ```
///
/// `foo` *should* be present among the completion variants. Filtering by
/// identifier prefix/fuzzy match should be done higher in the stack, together
/// with ordering of completions (currently this is done by the client).
pub(crate) fn completions(
    db: &RootDatabase,
    config: &CompletionConfig,
    position: FilePosition,
) -> Option<Completions> {
    let ctx = CompletionContext::new(db, position, config)?;

    let mut acc = Completions::default();
    complete_attribute::complete_attribute(&mut acc, &ctx);
    complete_fn_param::complete_fn_param(&mut acc, &ctx);
    complete_keyword::complete_expr_keyword(&mut acc, &ctx);
    complete_keyword::complete_use_tree_keyword(&mut acc, &ctx);
    complete_snippet::complete_expr_snippet(&mut acc, &ctx);
    complete_snippet::complete_item_snippet(&mut acc, &ctx);
    complete_qualified_path::complete_qualified_path(&mut acc, &ctx);
    complete_unqualified_path::complete_unqualified_path(&mut acc, &ctx);
    complete_dot::complete_dot(&mut acc, &ctx);
    complete_record::complete_record(&mut acc, &ctx);
    complete_pattern::complete_pattern(&mut acc, &ctx);
    complete_postfix::complete_postfix(&mut acc, &ctx);
    complete_macro_in_item_position::complete_macro_in_item_position(&mut acc, &ctx);
    complete_trait_impl::complete_trait_impl(&mut acc, &ctx);
    complete_mod::complete_mod(&mut acc, &ctx);

    Some(acc)
}

#[cfg(test)]
mod tests {
    use crate::completion::completion_config::CompletionConfig;
    use crate::mock_analysis::analysis_and_position;

    struct DetailAndDocumentation<'a> {
        detail: &'a str,
        documentation: &'a str,
    }

    fn check_detail_and_documentation(ra_fixture: &str, expected: DetailAndDocumentation) {
        let (analysis, position) = analysis_and_position(ra_fixture);
        let config = CompletionConfig::default();
        let completions = analysis.completions(&config, position).unwrap().unwrap();
        for item in completions {
            if item.detail() == Some(expected.detail) {
                let opt = item.documentation();
                let doc = opt.as_ref().map(|it| it.as_str());
                assert_eq!(doc, Some(expected.documentation));
                return;
            }
        }
        panic!("completion detail not found: {}", expected.detail)
    }

    #[test]
    fn test_completion_detail_from_macro_generated_struct_fn_doc_attr() {
        check_detail_and_documentation(
            r#"
            //- /lib.rs
            macro_rules! bar {
                () => {
                    struct Bar;
                    impl Bar {
                        #[doc = "Do the foo"]
                        fn foo(&self) {}
                    }
                }
            }

            bar!();

            fn foo() {
                let bar = Bar;
                bar.fo<|>;
            }
            "#,
            DetailAndDocumentation { detail: "fn foo(&self)", documentation: "Do the foo" },
        );
    }

    #[test]
    fn test_completion_detail_from_macro_generated_struct_fn_doc_comment() {
        check_detail_and_documentation(
            r#"
            //- /lib.rs
            macro_rules! bar {
                () => {
                    struct Bar;
                    impl Bar {
                        /// Do the foo
                        fn foo(&self) {}
                    }
                }
            }

            bar!();

            fn foo() {
                let bar = Bar;
                bar.fo<|>;
            }
            "#,
            DetailAndDocumentation { detail: "fn foo(&self)", documentation: " Do the foo" },
        );
    }
}
