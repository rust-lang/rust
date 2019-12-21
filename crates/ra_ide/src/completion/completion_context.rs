//! FIXME: write short doc here

use ra_syntax::{
    algo::{find_covering_element, find_node_at_offset},
    ast, AstNode, Parse, SourceFile,
    SyntaxKind::*,
    SyntaxNode, SyntaxToken, TextRange, TextUnit,
};
use ra_text_edit::AtomTextEdit;

use crate::{db, FilePosition};

/// `CompletionContext` is created early during completion to figure out, where
/// exactly is the cursor, syntax-wise.
#[derive(Debug)]
pub(crate) struct CompletionContext<'a> {
    pub(super) db: &'a db::RootDatabase,
    pub(super) analyzer: hir::SourceAnalyzer,
    pub(super) offset: TextUnit,
    pub(super) token: SyntaxToken,
    pub(super) module: Option<hir::Module>,
    pub(super) name_ref_syntax: Option<ast::NameRef>,
    pub(super) function_syntax: Option<ast::FnDef>,
    pub(super) use_item_syntax: Option<ast::UseItem>,
    pub(super) record_lit_syntax: Option<ast::RecordLit>,
    pub(super) record_lit_pat: Option<ast::RecordPat>,
    pub(super) is_param: bool,
    /// If a name-binding or reference to a const in a pattern.
    /// Irrefutable patterns (like let) are excluded.
    pub(super) is_pat_binding: bool,
    /// A single-indent path, like `foo`. `::foo` should not be considered a trivial path.
    pub(super) is_trivial_path: bool,
    /// If not a trivial path, the prefix (qualifier).
    pub(super) path_prefix: Option<hir::Path>,
    pub(super) after_if: bool,
    /// `true` if we are a statement or a last expr in the block.
    pub(super) can_be_stmt: bool,
    /// Something is typed at the "top" level, in module or impl/trait.
    pub(super) is_new_item: bool,
    /// The receiver if this is a field or method access, i.e. writing something.<|>
    pub(super) dot_receiver: Option<ast::Expr>,
    pub(super) dot_receiver_is_ambiguous_float_literal: bool,
    /// If this is a call (method or function) in particular, i.e. the () are already there.
    pub(super) is_call: bool,
    pub(super) is_path_type: bool,
    pub(super) has_type_args: bool,
}

impl<'a> CompletionContext<'a> {
    pub(super) fn new(
        db: &'a db::RootDatabase,
        original_parse: &'a Parse<ast::SourceFile>,
        position: FilePosition,
    ) -> Option<CompletionContext<'a>> {
        let src = hir::ModuleSource::from_position(db, position);
        let module = hir::Module::from_definition(
            db,
            hir::InFile { file_id: position.file_id.into(), value: src },
        );
        let token =
            original_parse.tree().syntax().token_at_offset(position.offset).left_biased()?;
        let analyzer = hir::SourceAnalyzer::new(
            db,
            hir::InFile::new(position.file_id.into(), &token.parent()),
            Some(position.offset),
        );
        let mut ctx = CompletionContext {
            db,
            analyzer,
            token,
            offset: position.offset,
            module,
            name_ref_syntax: None,
            function_syntax: None,
            use_item_syntax: None,
            record_lit_syntax: None,
            record_lit_pat: None,
            is_param: false,
            is_pat_binding: false,
            is_trivial_path: false,
            path_prefix: None,
            after_if: false,
            can_be_stmt: false,
            is_new_item: false,
            dot_receiver: None,
            is_call: false,
            is_path_type: false,
            has_type_args: false,
            dot_receiver_is_ambiguous_float_literal: false,
        };
        ctx.fill(&original_parse, position.offset);
        Some(ctx)
    }

    // The range of the identifier that is being completed.
    pub(crate) fn source_range(&self) -> TextRange {
        match self.token.kind() {
            // workaroud when completion is triggered by trigger characters.
            IDENT => self.token.text_range(),
            _ => TextRange::offset_len(self.offset, 0.into()),
        }
    }

    fn fill(&mut self, original_parse: &'a Parse<ast::SourceFile>, offset: TextUnit) {
        // Insert a fake ident to get a valid parse tree. We will use this file
        // to determine context, though the original_file will be used for
        // actual completion.
        let file = {
            let edit = AtomTextEdit::insert(offset, "intellijRulezz".to_string());
            original_parse.reparse(&edit).tree()
        };

        // First, let's try to complete a reference to some declaration.
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(file.syntax(), offset) {
            // Special case, `trait T { fn foo(i_am_a_name_ref) {} }`.
            // See RFC#1685.
            if is_node::<ast::Param>(name_ref.syntax()) {
                self.is_param = true;
                return;
            }
            self.classify_name_ref(original_parse.tree(), name_ref);
        }

        // Otherwise, see if this is a declaration. We can use heuristics to
        // suggest declaration names, see `CompletionKind::Magic`.
        if let Some(name) = find_node_at_offset::<ast::Name>(file.syntax(), offset) {
            if let Some(bind_pat) = name.syntax().ancestors().find_map(ast::BindPat::cast) {
                let parent = bind_pat.syntax().parent();
                if parent.clone().and_then(ast::MatchArm::cast).is_some()
                    || parent.and_then(ast::Condition::cast).is_some()
                {
                    self.is_pat_binding = true;
                }
            }
            if is_node::<ast::Param>(name.syntax()) {
                self.is_param = true;
                return;
            }
            if name.syntax().ancestors().find_map(ast::RecordFieldPatList::cast).is_some() {
                self.record_lit_pat =
                    find_node_at_offset(original_parse.tree().syntax(), self.offset);
            }
        }
    }

    fn classify_name_ref(&mut self, original_file: SourceFile, name_ref: ast::NameRef) {
        self.name_ref_syntax =
            find_node_at_offset(original_file.syntax(), name_ref.syntax().text_range().start());
        let name_range = name_ref.syntax().text_range();
        if name_ref.syntax().parent().and_then(ast::RecordField::cast).is_some() {
            self.record_lit_syntax = find_node_at_offset(original_file.syntax(), self.offset);
        }

        let top_node = name_ref
            .syntax()
            .ancestors()
            .take_while(|it| it.text_range() == name_range)
            .last()
            .unwrap();

        match top_node.parent().map(|it| it.kind()) {
            Some(SOURCE_FILE) | Some(ITEM_LIST) => {
                self.is_new_item = true;
                return;
            }
            _ => (),
        }

        self.use_item_syntax = self.token.parent().ancestors().find_map(ast::UseItem::cast);

        self.function_syntax = self
            .token
            .parent()
            .ancestors()
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::FnDef::cast);

        let parent = match name_ref.syntax().parent() {
            Some(it) => it,
            None => return,
        };

        if let Some(segment) = ast::PathSegment::cast(parent.clone()) {
            let path = segment.parent_path();
            self.is_call = path
                .syntax()
                .parent()
                .and_then(ast::PathExpr::cast)
                .and_then(|it| it.syntax().parent().and_then(ast::CallExpr::cast))
                .is_some();

            self.is_path_type = path.syntax().parent().and_then(ast::PathType::cast).is_some();
            self.has_type_args = segment.type_arg_list().is_some();

            if let Some(path) = hir::Path::from_ast(path.clone()) {
                if let Some(path_prefix) = path.qualifier() {
                    self.path_prefix = Some(path_prefix);
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
                        if let Some(stmt) = ast::ExprStmt::cast(node.clone()) {
                            return Some(
                                stmt.syntax().text_range() == name_ref.syntax().text_range(),
                            );
                        }
                        if let Some(block) = ast::Block::cast(node) {
                            return Some(
                                block.expr().map(|e| e.syntax().text_range())
                                    == Some(name_ref.syntax().text_range()),
                            );
                        }
                        None
                    })
                    .unwrap_or(false);

                if let Some(off) = name_ref.syntax().text_range().start().checked_sub(2.into()) {
                    if let Some(if_expr) =
                        find_node_at_offset::<ast::IfExpr>(original_file.syntax(), off)
                    {
                        if if_expr.syntax().text_range().end()
                            < name_ref.syntax().text_range().start()
                        {
                            self.after_if = true;
                        }
                    }
                }
            }
        }
        if let Some(field_expr) = ast::FieldExpr::cast(parent.clone()) {
            // The receiver comes before the point of insertion of the fake
            // ident, so it should have the same range in the non-modified file
            self.dot_receiver = field_expr
                .expr()
                .map(|e| e.syntax().text_range())
                .and_then(|r| find_node_with_range(original_file.syntax(), r));
            self.dot_receiver_is_ambiguous_float_literal =
                if let Some(ast::Expr::Literal(l)) = &self.dot_receiver {
                    match l.kind() {
                        ast::LiteralKind::FloatNumber { .. } => l.token().text().ends_with('.'),
                        _ => false,
                    }
                } else {
                    false
                }
        }
        if let Some(method_call_expr) = ast::MethodCallExpr::cast(parent) {
            // As above
            self.dot_receiver = method_call_expr
                .expr()
                .map(|e| e.syntax().text_range())
                .and_then(|r| find_node_with_range(original_file.syntax(), r));
            self.is_call = true;
        }
    }
}

fn find_node_with_range<N: AstNode>(syntax: &SyntaxNode, range: TextRange) -> Option<N> {
    find_covering_element(syntax, range).ancestors().find_map(N::cast)
}

fn is_node<N: AstNode>(node: &SyntaxNode) -> bool {
    match node.ancestors().find_map(N::cast) {
        None => false,
        Some(n) => n.syntax().text_range() == node.text_range(),
    }
}
