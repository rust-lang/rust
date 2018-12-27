use ra_editor::find_node_at_offset;
use ra_text_edit::AtomTextEdit;
use ra_syntax::{
    algo::{find_leaf_at_offset, find_covering_node},
    ast,
    AstNode,
    SyntaxNodeRef,
    SourceFileNode,
    TextUnit,
    TextRange,
    SyntaxKind::*,
};
use hir::source_binder;

use crate::{db, FilePosition, Cancelable};

/// `CompletionContext` is created early during completion to figure out, where
/// exactly is the cursor, syntax-wise.
#[derive(Debug)]
pub(super) struct CompletionContext<'a> {
    pub(super) db: &'a db::RootDatabase,
    pub(super) offset: TextUnit,
    pub(super) leaf: SyntaxNodeRef<'a>,
    pub(super) module: Option<hir::Module>,
    pub(super) function: Option<hir::Function>,
    pub(super) function_syntax: Option<ast::FnDef<'a>>,
    pub(super) is_param: bool,
    /// A single-indent path, like `foo`.
    pub(super) is_trivial_path: bool,
    /// If not a trivial, path, the prefix (qualifier).
    pub(super) path_prefix: Option<hir::Path>,
    pub(super) after_if: bool,
    pub(super) is_stmt: bool,
    /// Something is typed at the "top" level, in module or impl/trait.
    pub(super) is_new_item: bool,
    /// The receiver if this is a field or method access, i.e. writing something.<|>
    pub(super) dot_receiver: Option<ast::Expr<'a>>,
    /// If this is a method call in particular, i.e. the () are already there.
    pub(super) is_method_call: bool,
}

impl<'a> CompletionContext<'a> {
    pub(super) fn new(
        db: &'a db::RootDatabase,
        original_file: &'a SourceFileNode,
        position: FilePosition,
    ) -> Cancelable<Option<CompletionContext<'a>>> {
        let module = source_binder::module_from_position(db, position)?;
        let leaf =
            ctry!(find_leaf_at_offset(original_file.syntax(), position.offset).left_biased());
        let mut ctx = CompletionContext {
            db,
            leaf,
            offset: position.offset,
            module,
            function: None,
            function_syntax: None,
            is_param: false,
            is_trivial_path: false,
            path_prefix: None,
            after_if: false,
            is_stmt: false,
            is_new_item: false,
            dot_receiver: None,
            is_method_call: false,
        };
        ctx.fill(original_file, position.offset);
        Ok(Some(ctx))
    }

    fn fill(&mut self, original_file: &'a SourceFileNode, offset: TextUnit) {
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
            self.classify_name_ref(original_file, name_ref);
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
    fn classify_name_ref(&mut self, original_file: &'a SourceFileNode, name_ref: ast::NameRef) {
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

        self.function_syntax = self
            .leaf
            .ancestors()
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::FnDef::cast);
        match (&self.module, self.function_syntax) {
            (Some(module), Some(fn_def)) => {
                let function = source_binder::function_from_module(self.db, module, fn_def);
                self.function = Some(function);
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
                    if let Some(if_expr) =
                        find_node_at_offset::<ast::IfExpr>(original_file.syntax(), off)
                    {
                        if if_expr.syntax().range().end() < name_ref.syntax().range().start() {
                            self.after_if = true;
                        }
                    }
                }
            }
        }
        if let Some(field_expr) = ast::FieldExpr::cast(parent) {
            // The receiver comes before the point of insertion of the fake
            // ident, so it should have the same range in the non-modified file
            self.dot_receiver = field_expr
                .expr()
                .map(|e| e.syntax().range())
                .and_then(|r| find_node_with_range(original_file.syntax(), r));
        }
        if let Some(method_call_expr) = ast::MethodCallExpr::cast(parent) {
            // As above
            self.dot_receiver = method_call_expr
                .expr()
                .map(|e| e.syntax().range())
                .and_then(|r| find_node_with_range(original_file.syntax(), r));
            self.is_method_call = true;
        }
    }
}

fn find_node_with_range<'a, N: AstNode<'a>>(
    syntax: SyntaxNodeRef<'a>,
    range: TextRange,
) -> Option<N> {
    let node = find_covering_node(syntax, range);
    node.ancestors().find_map(N::cast)
}

fn is_node<'a, N: AstNode<'a>>(node: SyntaxNodeRef<'a>) -> bool {
    match node.ancestors().filter_map(N::cast).next() {
        None => false,
        Some(n) => n.syntax().range() == node.range(),
    }
}
