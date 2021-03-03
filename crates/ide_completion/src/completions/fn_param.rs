//! See `complete_fn_param`.

use rustc_hash::FxHashMap;
use syntax::{
    ast::{self, ModuleItemOwner},
    match_ast, AstNode,
};

use crate::{CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions};

/// Complete repeated parameters, both name and type. For example, if all
/// functions in a file have a `spam: &mut Spam` parameter, a completion with
/// `spam: &mut Spam` insert text/label and `spam` lookup string will be
/// suggested.
pub(crate) fn complete_fn_param(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_param {
        return;
    }

    let mut params = FxHashMap::default();

    let me = ctx.token.ancestors().find_map(ast::Fn::cast);
    let mut process_fn = |func: ast::Fn| {
        if Some(&func) == me.as_ref() {
            return;
        }
        func.param_list().into_iter().flat_map(|it| it.params()).for_each(|param| {
            if let Some(pat) = param.pat() {
                let text = param.syntax().text().to_string();
                let lookup = pat.syntax().text().to_string();
                params.entry(text).or_insert(lookup);
            }
        });
    };

    for node in ctx.token.parent().ancestors() {
        match_ast! {
            match node {
                ast::SourceFile(it) => it.items().filter_map(|item| match item {
                    ast::Item::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut process_fn),
                ast::ItemList(it) => it.items().filter_map(|item| match item {
                    ast::Item::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut process_fn),
                ast::AssocItemList(it) => it.assoc_items().filter_map(|item| match item {
                    ast::AssocItem::Fn(it) => Some(it),
                    _ => None,
                }).for_each(&mut process_fn),
                _ => continue,
            }
        };
    }

    params.into_iter().for_each(|(label, lookup)| {
        CompletionItem::new(CompletionKind::Magic, ctx.source_range(), label)
            .kind(CompletionItemKind::Binding)
            .lookup_by(lookup)
            .add_to(acc)
    });
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{test_utils::completion_list, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list(ra_fixture, CompletionKind::Magic);
        expect.assert_eq(&actual);
    }

    #[test]
    fn test_param_completion_last_param() {
        check(
            r#"
fn foo(file_id: FileId) {}
fn bar(file_id: FileId) {}
fn baz(file$0) {}
"#,
            expect![[r#"
                bn file_id: FileId
            "#]],
        );
    }

    #[test]
    fn test_param_completion_nth_param() {
        check(
            r#"
fn foo(file_id: FileId) {}
fn baz(file$0, x: i32) {}
"#,
            expect![[r#"
                bn file_id: FileId
            "#]],
        );
    }

    #[test]
    fn test_param_completion_trait_param() {
        check(
            r#"
pub(crate) trait SourceRoot {
    pub fn contains(&self, file_id: FileId) -> bool;
    pub fn module_map(&self) -> &ModuleMap;
    pub fn lines(&self, file_id: FileId) -> &LineIndex;
    pub fn syntax(&self, file$0)
}
"#,
            expect![[r#"
                bn file_id: FileId
            "#]],
        );
    }

    #[test]
    fn completes_param_in_inner_function() {
        check(
            r#"
fn outer(text: String) {
    fn inner($0)
}
"#,
            expect![[r#"
                bn text: String
            "#]],
        )
    }
}
