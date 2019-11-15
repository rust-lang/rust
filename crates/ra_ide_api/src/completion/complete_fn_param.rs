//! FIXME: write short doc here

use ra_syntax::{ast, match_ast, AstNode};
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
        match_ast! {
            match node {
                ast::SourceFile(it) => { process(it, &mut params) },
                ast::ItemList(it) => { process(it, &mut params) },
                _ => (),
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

    fn process<N: ast::FnDefOwner>(node: N, params: &mut FxHashMap<String, (u32, ast::Param)>) {
        node.functions().filter_map(|it| it.param_list()).flat_map(|it| it.params()).for_each(
            |param| {
                let text = param.syntax().text().to_string();
                params.entry(text).or_insert((0, param)).0 += 1;
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;

    fn do_magic_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Magic)
    }

    #[test]
    fn test_param_completion_last_param() {
        assert_debug_snapshot!(
        do_magic_completion(
                r"
                fn foo(file_id: FileId) {}
                fn bar(file_id: FileId) {}
                fn baz(file<|>) {}
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "file_id: FileId",
                source_range: [110; 114),
                delete: [110; 114),
                insert: "file_id: FileId",
                lookup: "file_id",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_param_completion_nth_param() {
        assert_debug_snapshot!(
        do_magic_completion(
                r"
                fn foo(file_id: FileId) {}
                fn bar(file_id: FileId) {}
                fn baz(file<|>, x: i32) {}
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "file_id: FileId",
                source_range: [110; 114),
                delete: [110; 114),
                insert: "file_id: FileId",
                lookup: "file_id",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_param_completion_trait_param() {
        assert_debug_snapshot!(
        do_magic_completion(
                r"
                pub(crate) trait SourceRoot {
                    pub fn contains(&self, file_id: FileId) -> bool;
                    pub fn module_map(&self) -> &ModuleMap;
                    pub fn lines(&self, file_id: FileId) -> &LineIndex;
                    pub fn syntax(&self, file<|>)
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "file_id: FileId",
                source_range: [289; 293),
                delete: [289; 293),
                insert: "file_id: FileId",
                lookup: "file_id",
            },
        ]
        "###
        );
    }
}
