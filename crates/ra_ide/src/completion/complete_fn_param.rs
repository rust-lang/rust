//! FIXME: write short doc here

use ra_syntax::{
    ast::{self, ModuleItemOwner},
    match_ast, AstNode,
};
use rustc_hash::FxHashMap;

use crate::completion::{CompletionContext, CompletionItem, CompletionKind, Completions};

/// Complete repeated parameters, both name and type. For example, if all
/// functions in a file have a `spam: &mut Spam` parameter, a completion with
/// `spam: &mut Spam` insert text/label and `spam` lookup string will be
/// suggested.
pub(super) fn complete_fn_param(acc: &mut Completions, ctx: &CompletionContext) {
    if !ctx.is_param {
        return;
    }

    let mut params = FxHashMap::default();
    for node in ctx.token.parent().ancestors() {
        let items = match_ast! {
            match node {
                ast::SourceFile(it) => it.items(),
                ast::ItemList(it) => it.items(),
                _ => continue,
            }
        };
        for item in items {
            if let ast::ModuleItem::FnDef(func) = item {
                func.param_list().into_iter().flat_map(|it| it.params()).for_each(|param| {
                    let text = param.syntax().text().to_string();
                    params.entry(text).or_insert((0, param)).0 += 1;
                })
            }
        }
    }
    params
        .into_iter()
        .filter_map(|(label, (count, param))| {
            let lookup = param.pat()?.syntax().text().to_string();
            if count < 2 {
                None
            } else {
                Some((label, lookup))
            }
        })
        .for_each(|(label, lookup)| {
            CompletionItem::new(CompletionKind::Magic, ctx.source_range(), label)
                .lookup_by(lookup)
                .add_to(acc)
        });
}

#[cfg(test)]
mod tests {
    use expect::{expect, Expect};

    use crate::completion::{test_utils::do_completion, CompletionKind};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = do_completion(ra_fixture, CompletionKind::Magic);
        expect.assert_debug_eq(&actual);
    }

    #[test]
    fn test_param_completion_last_param() {
        check(
            r#"
fn foo(file_id: FileId) {}
fn bar(file_id: FileId) {}
fn baz(file<|>) {}
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "file_id: FileId",
                        source_range: 61..65,
                        delete: 61..65,
                        insert: "file_id: FileId",
                        lookup: "file_id",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn test_param_completion_nth_param() {
        check(
            r#"
fn foo(file_id: FileId) {}
fn bar(file_id: FileId) {}
fn baz(file<|>, x: i32) {}
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "file_id: FileId",
                        source_range: 61..65,
                        delete: 61..65,
                        insert: "file_id: FileId",
                        lookup: "file_id",
                    },
                ]
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
    pub fn syntax(&self, file<|>)
}
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "file_id: FileId",
                        source_range: 208..212,
                        delete: 208..212,
                        insert: "file_id: FileId",
                        lookup: "file_id",
                    },
                ]
            "#]],
        );
    }
}
