mod completion_item;

mod complete_fn_param;
mod complete_keyword;
mod complete_snippet;
mod complete_path;
mod complete_scope;

use ra_editor::find_node_at_offset;
use ra_text_edit::AtomTextEdit;
use ra_syntax::{
    algo::find_leaf_at_offset,
    ast,
    AstNode,
    SyntaxNodeRef,
    SourceFileNode,
    TextUnit,
    SyntaxKind::*,
};
use ra_db::SyntaxDatabase;
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
    let ctx = ctry!(SyntaxContext::new(db, &original_file, position)?);

    let mut acc = Completions::default();

    complete_fn_param::complete_fn_param(&mut acc, &ctx);
    complete_keyword::complete_expr_keyword(&mut acc, &ctx);
    complete_snippet::complete_expr_snippet(&mut acc, &ctx);
    complete_snippet::complete_item_snippet(&mut acc, &ctx);
    complete_path::complete_path(&mut acc, &ctx)?;
    complete_scope::complete_scope(&mut acc, &ctx)?;

    Ok(Some(acc))
}

/// `SyntaxContext` is created early during completion to figure out, where
/// exactly is the cursor, syntax-wise.
#[derive(Debug)]
pub(super) struct SyntaxContext<'a> {
    db: &'a db::RootDatabase,
    offset: TextUnit,
    leaf: SyntaxNodeRef<'a>,
    module: Option<hir::Module>,
    enclosing_fn: Option<ast::FnDef<'a>>,
    is_param: bool,
    /// A single-indent path, like `foo`.
    is_trivial_path: bool,
    /// If not a trivial, path, the prefix (qualifier).
    path_prefix: Option<hir::Path>,
    after_if: bool,
    is_stmt: bool,
    /// Something is typed at the "top" level, in module or impl/trait.
    is_new_item: bool,
}

impl<'a> SyntaxContext<'a> {
    pub(super) fn new(
        db: &'a db::RootDatabase,
        original_file: &'a SourceFileNode,
        position: FilePosition,
    ) -> Cancelable<Option<SyntaxContext<'a>>> {
        let module = source_binder::module_from_position(db, position)?;
        let leaf =
            ctry!(find_leaf_at_offset(original_file.syntax(), position.offset).left_biased());
        let mut ctx = SyntaxContext {
            db,
            leaf,
            offset: position.offset,
            module,
            enclosing_fn: None,
            is_param: false,
            is_trivial_path: false,
            path_prefix: None,
            after_if: false,
            is_stmt: false,
            is_new_item: false,
        };
        ctx.fill(original_file, position.offset);
        Ok(Some(ctx))
    }

    fn fill(&mut self, original_file: &SourceFileNode, offset: TextUnit) {
        // Insert a fake ident to get a valid parse tree. We will use this file
        // to determine context, though the original_file will be used for
        // actual completion.
        let file = {
            let edit = AtomTextEdit::insert(offset, "intellijRulezz".to_string());
            original_file.reparse(&edit)
        };

        // First, let's try to complete a reference to some declaration.
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(file.syntax(), offset) {
            // Special case, `trait T { fn foo(i_am_a_name_ref) {} }`.
            // See RFC#1685.
            if is_node::<ast::Param>(name_ref.syntax()) {
                self.is_param = true;
                return;
            }
            self.classify_name_ref(&file, name_ref);
        }

        // Otherwise, see if this is a declaration. We can use heuristics to
        // suggest declaration names, see `CompletionKind::Magic`.
        if let Some(name) = find_node_at_offset::<ast::Name>(file.syntax(), offset) {
            if is_node::<ast::Param>(name.syntax()) {
                self.is_param = true;
                return;
            }
        }
    }
    fn classify_name_ref(&mut self, file: &SourceFileNode, name_ref: ast::NameRef) {
        let name_range = name_ref.syntax().range();
        let top_node = name_ref
            .syntax()
            .ancestors()
            .take_while(|it| it.range() == name_range)
            .last()
            .unwrap();

        match top_node.parent().map(|it| it.kind()) {
            Some(SOURCE_FILE) | Some(ITEM_LIST) => {
                self.is_new_item = true;
                return;
            }
            _ => (),
        }

        let parent = match name_ref.syntax().parent() {
            Some(it) => it,
            None => return,
        };
        if let Some(segment) = ast::PathSegment::cast(parent) {
            let path = segment.parent_path();
            if let Some(mut path) = hir::Path::from_ast(path) {
                if !path.is_ident() {
                    path.segments.pop().unwrap();
                    self.path_prefix = Some(path);
                    return;
                }
            }
            if path.qualifier().is_none() {
                self.is_trivial_path = true;
                self.enclosing_fn = self
                    .leaf
                    .ancestors()
                    .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
                    .find_map(ast::FnDef::cast);

                self.is_stmt = match name_ref
                    .syntax()
                    .ancestors()
                    .filter_map(ast::ExprStmt::cast)
                    .next()
                {
                    None => false,
                    Some(expr_stmt) => expr_stmt.syntax().range() == name_ref.syntax().range(),
                };

                if let Some(off) = name_ref.syntax().range().start().checked_sub(2.into()) {
                    if let Some(if_expr) = find_node_at_offset::<ast::IfExpr>(file.syntax(), off) {
                        if if_expr.syntax().range().end() < name_ref.syntax().range().start() {
                            self.after_if = true;
                        }
                    }
                }
            }
        }
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
