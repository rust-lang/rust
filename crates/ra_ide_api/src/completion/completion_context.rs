use ra_text_edit::AtomTextEdit;
use ra_syntax::{
    AstNode, SyntaxNode, SourceFile, TextUnit, TextRange,
    ast,
    algo::{find_leaf_at_offset, find_covering_node, find_node_at_offset},
    SyntaxKind::*,
};
use hir::{source_binder, Resolver};

use crate::{db, FilePosition};

/// `CompletionContext` is created early during completion to figure out, where
/// exactly is the cursor, syntax-wise.
#[derive(Debug)]
pub(crate) struct CompletionContext<'a> {
    pub(super) db: &'a db::RootDatabase,
    pub(super) offset: TextUnit,
    pub(super) leaf: &'a SyntaxNode,
    pub(super) resolver: Resolver,
    pub(super) module: Option<hir::Module>,
    pub(super) function: Option<hir::Function>,
    pub(super) function_syntax: Option<&'a ast::FnDef>,
    pub(super) use_item_syntax: Option<&'a ast::UseItem>,
    pub(super) struct_lit_syntax: Option<&'a ast::StructLit>,
    pub(super) is_param: bool,
    /// If a name-binding or reference to a const in a pattern.
    /// Irrefutable patterns (like let) are excluded.
    pub(super) is_pat_binding: bool,
    /// A single-indent path, like `foo`. `::foo` should not be considered a trivial path.
    pub(super) is_trivial_path: bool,
    /// If not a trivial, path, the prefix (qualifier).
    pub(super) path_prefix: Option<hir::Path>,
    pub(super) after_if: bool,
    /// `true` if we are a statement or a last expr in the block.
    pub(super) can_be_stmt: bool,
    /// Something is typed at the "top" level, in module or impl/trait.
    pub(super) is_new_item: bool,
    /// The receiver if this is a field or method access, i.e. writing something.<|>
    pub(super) dot_receiver: Option<&'a ast::Expr>,
    /// If this is a call (method or function) in particular, i.e. the () are already there.
    pub(super) is_call: bool,
}

impl<'a> CompletionContext<'a> {
    pub(super) fn new(
        db: &'a db::RootDatabase,
        original_file: &'a SourceFile,
        position: FilePosition,
    ) -> Option<CompletionContext<'a>> {
        let resolver = source_binder::resolver_for_position(db, position);
        let module = source_binder::module_from_position(db, position);
        let leaf = find_leaf_at_offset(original_file.syntax(), position.offset).left_biased()?;
        let mut ctx = CompletionContext {
            db,
            leaf,
            offset: position.offset,
            resolver,
            module,
            function: None,
            function_syntax: None,
            use_item_syntax: None,
            struct_lit_syntax: None,
            is_param: false,
            is_pat_binding: false,
            is_trivial_path: false,
            path_prefix: None,
            after_if: false,
            can_be_stmt: false,
            is_new_item: false,
            dot_receiver: None,
            is_call: false,
        };
        ctx.fill(original_file, position.offset);
        Some(ctx)
    }

    // The range of the identifier that is being completed.
    pub(crate) fn source_range(&self) -> TextRange {
        match self.leaf.kind() {
            // workaroud when completion is triggered by trigger characters.
            IDENT => self.leaf.range(),
            _ => TextRange::offset_len(self.offset, 0.into()),
        }
    }

    fn fill(&mut self, original_file: &'a SourceFile, offset: TextUnit) {
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
            if is_node::<ast::BindPat>(name.syntax()) {
                let bind_pat = name.syntax().ancestors().find_map(ast::BindPat::cast).unwrap();
                let parent = bind_pat.syntax().parent();
                if parent.and_then(ast::MatchArm::cast).is_some()
                    || parent.and_then(ast::Condition::cast).is_some()
                {
                    self.is_pat_binding = true;
                }
            }
            if is_node::<ast::Param>(name.syntax()) {
                self.is_param = true;
                return;
            }
        }
    }

    fn classify_name_ref(&mut self, original_file: &'a SourceFile, name_ref: &ast::NameRef) {
        let name_range = name_ref.syntax().range();
        if name_ref.syntax().parent().and_then(ast::NamedField::cast).is_some() {
            self.struct_lit_syntax = find_node_at_offset(original_file.syntax(), self.offset);
        }

        let top_node =
            name_ref.syntax().ancestors().take_while(|it| it.range() == name_range).last().unwrap();

        match top_node.parent().map(|it| it.kind()) {
            Some(SOURCE_FILE) | Some(ITEM_LIST) => {
                self.is_new_item = true;
                return;
            }
            _ => (),
        }

        self.use_item_syntax = self.leaf.ancestors().find_map(ast::UseItem::cast);

        self.function_syntax = self
            .leaf
            .ancestors()
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::FnDef::cast);
        if let (Some(module), Some(fn_def)) = (self.module, self.function_syntax) {
            let function = source_binder::function_from_module(self.db, module, fn_def);
            self.function = Some(function);
        }

        let parent = match name_ref.syntax().parent() {
            Some(it) => it,
            None => return,
        };
        if let Some(segment) = ast::PathSegment::cast(parent) {
            let path = segment.parent_path();
            self.is_call = path
                .syntax()
                .parent()
                .and_then(ast::PathExpr::cast)
                .and_then(|it| it.syntax().parent().and_then(ast::CallExpr::cast))
                .is_some();

            if let Some(mut path) = hir::Path::from_ast(path) {
                if !path.is_ident() {
                    path.segments.pop().unwrap();
                    self.path_prefix = Some(path);
                    return;
                }
            }
            if path.qualifier().is_none() {
                self.is_trivial_path = true;

                // Find either enclosing expr statement (thing with `;`) or a
                // block. If block, check that we are the last expr.
                self.can_be_stmt = name_ref
                    .syntax()
                    .ancestors()
                    .find_map(|node| {
                        if let Some(stmt) = ast::ExprStmt::cast(node) {
                            return Some(stmt.syntax().range() == name_ref.syntax().range());
                        }
                        if let Some(block) = ast::Block::cast(node) {
                            return Some(
                                block.expr().map(|e| e.syntax().range())
                                    == Some(name_ref.syntax().range()),
                            );
                        }
                        None
                    })
                    .unwrap_or(false);

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
            self.is_call = true;
        }
    }
}

fn find_node_with_range<N: AstNode>(syntax: &SyntaxNode, range: TextRange) -> Option<&N> {
    let node = find_covering_node(syntax, range);
    node.ancestors().find_map(N::cast)
}

fn is_node<N: AstNode>(node: &SyntaxNode) -> bool {
    match node.ancestors().filter_map(N::cast).next() {
        None => false,
        Some(n) => n.syntax().range() == node.range(),
    }
}
