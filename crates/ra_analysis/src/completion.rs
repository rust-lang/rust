mod completion_item;
mod reference_completion;

use ra_editor::find_node_at_offset;
use ra_text_edit::AtomTextEdit;
use ra_syntax::{
    algo::visit::{visitor_ctx, VisitorCtx},
    ast,
    AstNode,
    SyntaxNodeRef,
};
use ra_db::SyntaxDatabase;
use rustc_hash::{FxHashMap};
use hir::source_binder;

use crate::{
    db,
    Cancelable, FilePosition,
    completion::completion_item::{Completions, CompletionKind},
};

pub use crate::completion::completion_item::{CompletionItem, InsertText};

pub(crate) fn completions(
    db: &db::RootDatabase,
    position: FilePosition,
) -> Cancelable<Option<Completions>> {
    let original_file = db.source_file(position.file_id);
    // Insert a fake ident to get a valid parse tree
    let file = {
        let edit = AtomTextEdit::insert(position.offset, "intellijRulezz".to_string());
        original_file.reparse(&edit)
    };

    let module = ctry!(source_binder::module_from_position(db, position)?);

    let mut acc = Completions::default();
    let mut has_completions = false;
    // First, let's try to complete a reference to some declaration.
    if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(file.syntax(), position.offset) {
        has_completions = true;
        reference_completion::completions(&mut acc, db, &module, &file, name_ref)?;
        // special case, `trait T { fn foo(i_am_a_name_ref) {} }`
        if is_node::<ast::Param>(name_ref.syntax()) {
            param_completions(&mut acc, name_ref.syntax());
        }
    }

    // Otherwise, if this is a declaration, use heuristics to suggest a name.
    if let Some(name) = find_node_at_offset::<ast::Name>(file.syntax(), position.offset) {
        if is_node::<ast::Param>(name.syntax()) {
            has_completions = true;
            param_completions(&mut acc, name.syntax());
        }
    }
    if !has_completions {
        return Ok(None);
    }
    Ok(Some(acc))
}

/// Complete repeated parametes, both name and type. For example, if all
/// functions in a file have a `spam: &mut Spam` parameter, a completion with
/// `spam: &mut Spam` insert text/label and `spam` lookup string will be
/// suggested.
fn param_completions(acc: &mut Completions, ctx: SyntaxNodeRef) {
    let mut params = FxHashMap::default();
    for node in ctx.ancestors() {
        let _ = visitor_ctx(&mut params)
            .visit::<ast::SourceFile, _>(process)
            .visit::<ast::ItemList, _>(process)
            .accept(node);
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
            CompletionItem::new(label)
                .lookup_by(lookup)
                .kind(CompletionKind::Magic)
                .add_to(acc)
        });

    fn process<'a, N: ast::FnDefOwner<'a>>(
        node: N,
        params: &mut FxHashMap<String, (u32, ast::Param<'a>)>,
    ) {
        node.functions()
            .filter_map(|it| it.param_list())
            .flat_map(|it| it.params())
            .for_each(|param| {
                let text = param.syntax().text().to_string();
                params.entry(text).or_insert((0, param)).0 += 1;
            })
    }
}

fn is_node<'a, N: AstNode<'a>>(node: SyntaxNodeRef<'a>) -> bool {
    match node.ancestors().filter_map(N::cast).next() {
        None => false,
        Some(n) => n.syntax().range() == node.range(),
    }
}

#[cfg(test)]
fn check_completion(code: &str, expected_completions: &str, kind: CompletionKind) {
    use crate::mock_analysis::{single_file_with_position, analysis_and_position};
    let (analysis, position) = if code.contains("//-") {
        analysis_and_position(code)
    } else {
        single_file_with_position(code)
    };
    let completions = completions(&analysis.imp.db, position).unwrap().unwrap();
    completions.assert_match(expected_completions, kind);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_magic_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Magic);
    }

    #[test]
    fn test_param_completion_last_param() {
        check_magic_completion(
            r"
            fn foo(file_id: FileId) {}
            fn bar(file_id: FileId) {}
            fn baz(file<|>) {}
            ",
            r#"file_id "file_id: FileId""#,
        );
    }

    #[test]
    fn test_param_completion_nth_param() {
        check_magic_completion(
            r"
            fn foo(file_id: FileId) {}
            fn bar(file_id: FileId) {}
            fn baz(file<|>, x: i32) {}
            ",
            r#"file_id "file_id: FileId""#,
        );
    }

    #[test]
    fn test_param_completion_trait_param() {
        check_magic_completion(
            r"
            pub(crate) trait SourceRoot {
                pub fn contains(&self, file_id: FileId) -> bool;
                pub fn module_map(&self) -> &ModuleMap;
                pub fn lines(&self, file_id: FileId) -> &LineIndex;
                pub fn syntax(&self, file<|>)
            }
            ",
            r#"file_id "file_id: FileId""#,
        );
    }
}
