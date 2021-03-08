//! See `CompletionContext` structure.

use hir::{Local, ScopeDef, Semantics, SemanticsScope, Type};
use ide_db::base_db::{FilePosition, SourceDatabase};
use ide_db::{call_info::ActiveParameter, RootDatabase};
use syntax::{
    algo::find_node_at_offset, ast, match_ast, AstNode, NodeOrToken, SyntaxKind::*, SyntaxNode,
    SyntaxToken, TextRange, TextSize,
};

use text_edit::Indel;

use crate::{
    patterns::{
        fn_is_prev, for_is_prev2, has_bind_pat_parent, has_block_expr_parent,
        has_field_list_parent, has_impl_as_prev_sibling, has_impl_parent,
        has_item_list_or_source_file_parent, has_ref_parent, has_trait_as_prev_sibling,
        has_trait_parent, if_is_prev, inside_impl_trait_block, is_in_loop_body, is_match_arm,
        unsafe_is_prev,
    },
    CompletionConfig,
};

/// `CompletionContext` is created early during completion to figure out, where
/// exactly is the cursor, syntax-wise.
#[derive(Debug)]
pub(crate) struct CompletionContext<'a> {
    pub(super) sema: Semantics<'a, RootDatabase>,
    pub(super) scope: SemanticsScope<'a>,
    pub(super) db: &'a RootDatabase,
    pub(super) config: &'a CompletionConfig,
    pub(super) position: FilePosition,
    /// The token before the cursor, in the original file.
    pub(super) original_token: SyntaxToken,
    /// The token before the cursor, in the macro-expanded file.
    pub(super) token: SyntaxToken,
    pub(super) krate: Option<hir::Crate>,
    pub(super) expected_type: Option<Type>,
    pub(super) name_ref_syntax: Option<ast::NameRef>,
    pub(super) function_syntax: Option<ast::Fn>,
    pub(super) use_item_syntax: Option<ast::Use>,
    pub(super) record_lit_syntax: Option<ast::RecordExpr>,
    pub(super) record_pat_syntax: Option<ast::RecordPat>,
    pub(super) record_field_syntax: Option<ast::RecordExprField>,
    pub(super) impl_def: Option<ast::Impl>,
    /// FIXME: `ActiveParameter` is string-based, which is very very wrong
    pub(super) active_parameter: Option<ActiveParameter>,
    pub(super) is_param: bool,
    /// If a name-binding or reference to a const in a pattern.
    /// Irrefutable patterns (like let) are excluded.
    pub(super) is_pat_binding_or_const: bool,
    pub(super) is_irrefutable_pat_binding: bool,
    /// A single-indent path, like `foo`. `::foo` should not be considered a trivial path.
    pub(super) is_trivial_path: bool,
    /// If not a trivial path, the prefix (qualifier).
    pub(super) path_qual: Option<ast::Path>,
    pub(super) after_if: bool,
    /// `true` if we are a statement or a last expr in the block.
    pub(super) can_be_stmt: bool,
    /// `true` if we expect an expression at the cursor position.
    pub(super) is_expr: bool,
    /// Something is typed at the "top" level, in module or impl/trait.
    pub(super) is_new_item: bool,
    /// The receiver if this is a field or method access, i.e. writing something.$0
    pub(super) dot_receiver: Option<ast::Expr>,
    pub(super) dot_receiver_is_ambiguous_float_literal: bool,
    /// If this is a call (method or function) in particular, i.e. the () are already there.
    pub(super) is_call: bool,
    /// Like `is_call`, but for tuple patterns.
    pub(super) is_pattern_call: bool,
    /// If this is a macro call, i.e. the () are already there.
    pub(super) is_macro_call: bool,
    pub(super) is_path_type: bool,
    pub(super) has_type_args: bool,
    pub(super) attribute_under_caret: Option<ast::Attr>,
    pub(super) mod_declaration_under_caret: Option<ast::Module>,
    pub(super) unsafe_is_prev: bool,
    pub(super) if_is_prev: bool,
    pub(super) block_expr_parent: bool,
    pub(super) bind_pat_parent: bool,
    pub(super) ref_pat_parent: bool,
    pub(super) in_loop_body: bool,
    pub(super) has_trait_parent: bool,
    pub(super) has_impl_parent: bool,
    pub(super) inside_impl_trait_block: bool,
    pub(super) has_field_list_parent: bool,
    pub(super) trait_as_prev_sibling: bool,
    pub(super) impl_as_prev_sibling: bool,
    pub(super) is_match_arm: bool,
    pub(super) has_item_list_or_source_file_parent: bool,
    pub(super) for_is_prev2: bool,
    pub(super) fn_is_prev: bool,
    pub(super) incomplete_let: bool,
    pub(super) locals: Vec<(String, Local)>,
}

impl<'a> CompletionContext<'a> {
    pub(super) fn new(
        db: &'a RootDatabase,
        position: FilePosition,
        config: &'a CompletionConfig,
    ) -> Option<CompletionContext<'a>> {
        let sema = Semantics::new(db);

        let original_file = sema.parse(position.file_id);

        // Insert a fake ident to get a valid parse tree. We will use this file
        // to determine context, though the original_file will be used for
        // actual completion.
        let file_with_fake_ident = {
            let parse = db.parse(position.file_id);
            let edit = Indel::insert(position.offset, "intellijRulezz".to_string());
            parse.reparse(&edit).tree()
        };
        let fake_ident_token =
            file_with_fake_ident.syntax().token_at_offset(position.offset).right_biased().unwrap();

        let krate = sema.to_module_def(position.file_id).map(|m| m.krate());
        let original_token =
            original_file.syntax().token_at_offset(position.offset).left_biased()?;
        let token = sema.descend_into_macros(original_token.clone());
        let scope = sema.scope_at_offset(&token.parent(), position.offset);
        let mut locals = vec![];
        scope.process_all_names(&mut |name, scope| {
            if let ScopeDef::Local(local) = scope {
                locals.push((name.to_string(), local));
            }
        });
        let mut ctx = CompletionContext {
            sema,
            scope,
            db,
            config,
            position,
            original_token,
            token,
            krate,
            expected_type: None,
            name_ref_syntax: None,
            function_syntax: None,
            use_item_syntax: None,
            record_lit_syntax: None,
            record_pat_syntax: None,
            record_field_syntax: None,
            impl_def: None,
            active_parameter: ActiveParameter::at(db, position),
            is_param: false,
            is_pat_binding_or_const: false,
            is_irrefutable_pat_binding: false,
            is_trivial_path: false,
            path_qual: None,
            after_if: false,
            can_be_stmt: false,
            is_expr: false,
            is_new_item: false,
            dot_receiver: None,
            dot_receiver_is_ambiguous_float_literal: false,
            is_call: false,
            is_pattern_call: false,
            is_macro_call: false,
            is_path_type: false,
            has_type_args: false,
            attribute_under_caret: None,
            mod_declaration_under_caret: None,
            unsafe_is_prev: false,
            if_is_prev: false,
            block_expr_parent: false,
            bind_pat_parent: false,
            ref_pat_parent: false,
            in_loop_body: false,
            has_trait_parent: false,
            has_impl_parent: false,
            inside_impl_trait_block: false,
            has_field_list_parent: false,
            trait_as_prev_sibling: false,
            impl_as_prev_sibling: false,
            is_match_arm: false,
            has_item_list_or_source_file_parent: false,
            for_is_prev2: false,
            fn_is_prev: false,
            incomplete_let: false,
            locals,
        };

        let mut original_file = original_file.syntax().clone();
        let mut hypothetical_file = file_with_fake_ident.syntax().clone();
        let mut offset = position.offset;
        let mut fake_ident_token = fake_ident_token;

        // Are we inside a macro call?
        while let (Some(actual_macro_call), Some(macro_call_with_fake_ident)) = (
            find_node_at_offset::<ast::MacroCall>(&original_file, offset),
            find_node_at_offset::<ast::MacroCall>(&hypothetical_file, offset),
        ) {
            if actual_macro_call.path().as_ref().map(|s| s.syntax().text())
                != macro_call_with_fake_ident.path().as_ref().map(|s| s.syntax().text())
            {
                break;
            }
            let hypothetical_args = match macro_call_with_fake_ident.token_tree() {
                Some(tt) => tt,
                None => break,
            };
            if let (Some(actual_expansion), Some(hypothetical_expansion)) = (
                ctx.sema.expand(&actual_macro_call),
                ctx.sema.speculative_expand(
                    &actual_macro_call,
                    &hypothetical_args,
                    fake_ident_token,
                ),
            ) {
                let new_offset = hypothetical_expansion.1.text_range().start();
                if new_offset > actual_expansion.text_range().end() {
                    break;
                }
                original_file = actual_expansion;
                hypothetical_file = hypothetical_expansion.0;
                fake_ident_token = hypothetical_expansion.1;
                offset = new_offset;
            } else {
                break;
            }
        }
        ctx.fill_keyword_patterns(&hypothetical_file, offset);
        ctx.fill(&original_file, hypothetical_file, offset);
        Some(ctx)
    }

    /// Checks whether completions in that particular case don't make much sense.
    /// Examples:
    /// - `fn $0` -- we expect function name, it's unlikely that "hint" will be helpful.
    ///   Exception for this case is `impl Trait for Foo`, where we would like to hint trait method names.
    /// - `for _ i$0` -- obviously, it'll be "in" keyword.
    pub(crate) fn no_completion_required(&self) -> bool {
        (self.fn_is_prev && !self.inside_impl_trait_block) || self.for_is_prev2
    }

    /// The range of the identifier that is being completed.
    pub(crate) fn source_range(&self) -> TextRange {
        // check kind of macro-expanded token, but use range of original token
        let kind = self.token.kind();
        if kind == IDENT || kind == UNDERSCORE || kind.is_keyword() {
            cov_mark::hit!(completes_if_prefix_is_keyword);
            self.original_token.text_range()
        } else {
            TextRange::empty(self.position.offset)
        }
    }

    fn fill_keyword_patterns(&mut self, file_with_fake_ident: &SyntaxNode, offset: TextSize) {
        let fake_ident_token = file_with_fake_ident.token_at_offset(offset).right_biased().unwrap();
        let syntax_element = NodeOrToken::Token(fake_ident_token);
        self.block_expr_parent = has_block_expr_parent(syntax_element.clone());
        self.unsafe_is_prev = unsafe_is_prev(syntax_element.clone());
        self.if_is_prev = if_is_prev(syntax_element.clone());
        self.bind_pat_parent = has_bind_pat_parent(syntax_element.clone());
        self.ref_pat_parent = has_ref_parent(syntax_element.clone());
        self.in_loop_body = is_in_loop_body(syntax_element.clone());
        self.has_trait_parent = has_trait_parent(syntax_element.clone());
        self.has_impl_parent = has_impl_parent(syntax_element.clone());
        self.inside_impl_trait_block = inside_impl_trait_block(syntax_element.clone());
        self.has_field_list_parent = has_field_list_parent(syntax_element.clone());
        self.impl_as_prev_sibling = has_impl_as_prev_sibling(syntax_element.clone());
        self.trait_as_prev_sibling = has_trait_as_prev_sibling(syntax_element.clone());
        self.is_match_arm = is_match_arm(syntax_element.clone());
        self.has_item_list_or_source_file_parent =
            has_item_list_or_source_file_parent(syntax_element.clone());
        self.mod_declaration_under_caret =
            find_node_at_offset::<ast::Module>(&file_with_fake_ident, offset)
                .filter(|module| module.item_list().is_none());
        self.for_is_prev2 = for_is_prev2(syntax_element.clone());
        self.fn_is_prev = fn_is_prev(syntax_element.clone());
        self.incomplete_let =
            syntax_element.ancestors().take(6).find_map(ast::LetStmt::cast).map_or(false, |it| {
                it.syntax().text_range().end() == syntax_element.text_range().end()
            });
    }

    fn fill_impl_def(&mut self) {
        self.impl_def = self
            .sema
            .ancestors_with_macros(self.token.parent())
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::Impl::cast);
    }

    fn fill(
        &mut self,
        original_file: &SyntaxNode,
        file_with_fake_ident: SyntaxNode,
        offset: TextSize,
    ) {
        // FIXME: this is wrong in at least two cases:
        //  * when there's no token `foo($0)`
        //  * when there is a token, but it happens to have type of it's own
        self.expected_type = self
            .token
            .ancestors()
            .find_map(|node| {
                let ty = match_ast! {
                    match node {
                        ast::Pat(it) => self.sema.type_of_pat(&it),
                        ast::Expr(it) => self.sema.type_of_expr(&it),
                        _ => return None,
                    }
                };
                Some(ty)
            })
            .flatten();
        self.attribute_under_caret = find_node_at_offset(&file_with_fake_ident, offset);

        // First, let's try to complete a reference to some declaration.
        if let Some(name_ref) = find_node_at_offset::<ast::NameRef>(&file_with_fake_ident, offset) {
            // Special case, `trait T { fn foo(i_am_a_name_ref) {} }`.
            // See RFC#1685.
            if is_node::<ast::Param>(name_ref.syntax()) {
                self.is_param = true;
                return;
            }
            // FIXME: remove this (V) duplication and make the check more precise
            if name_ref.syntax().ancestors().find_map(ast::RecordPatFieldList::cast).is_some() {
                self.record_pat_syntax =
                    self.sema.find_node_at_offset_with_macros(&original_file, offset);
            }
            self.classify_name_ref(original_file, name_ref, offset);
        }

        // Otherwise, see if this is a declaration. We can use heuristics to
        // suggest declaration names, see `CompletionKind::Magic`.
        if let Some(name) = find_node_at_offset::<ast::Name>(&file_with_fake_ident, offset) {
            if let Some(bind_pat) = name.syntax().ancestors().find_map(ast::IdentPat::cast) {
                self.is_pat_binding_or_const = true;
                if bind_pat.at_token().is_some()
                    || bind_pat.ref_token().is_some()
                    || bind_pat.mut_token().is_some()
                {
                    self.is_pat_binding_or_const = false;
                }
                if bind_pat.syntax().parent().and_then(ast::RecordPatFieldList::cast).is_some() {
                    self.is_pat_binding_or_const = false;
                }
                if let Some(Some(pat)) = bind_pat.syntax().ancestors().find_map(|node| {
                    match_ast! {
                        match node {
                            ast::LetStmt(it) => Some(it.pat()),
                            ast::Param(it) => Some(it.pat()),
                            _ => None,
                        }
                    }
                }) {
                    if pat.syntax().text_range().contains_range(bind_pat.syntax().text_range()) {
                        self.is_pat_binding_or_const = false;
                        self.is_irrefutable_pat_binding = true;
                    }
                }

                self.fill_impl_def();
            }
            if is_node::<ast::Param>(name.syntax()) {
                self.is_param = true;
                return;
            }
            // FIXME: remove this (^) duplication and make the check more precise
            if name.syntax().ancestors().find_map(ast::RecordPatFieldList::cast).is_some() {
                self.record_pat_syntax =
                    self.sema.find_node_at_offset_with_macros(&original_file, offset);
            }
        }
    }

    fn classify_name_ref(
        &mut self,
        original_file: &SyntaxNode,
        name_ref: ast::NameRef,
        offset: TextSize,
    ) {
        self.name_ref_syntax =
            find_node_at_offset(&original_file, name_ref.syntax().text_range().start());
        let name_range = name_ref.syntax().text_range();
        if ast::RecordExprField::for_field_name(&name_ref).is_some() {
            self.record_lit_syntax =
                self.sema.find_node_at_offset_with_macros(&original_file, offset);
        }

        self.fill_impl_def();

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

        self.use_item_syntax =
            self.sema.ancestors_with_macros(self.token.parent()).find_map(ast::Use::cast);

        self.function_syntax = self
            .sema
            .ancestors_with_macros(self.token.parent())
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::Fn::cast);

        self.record_field_syntax = self
            .sema
            .ancestors_with_macros(self.token.parent())
            .take_while(|it| {
                it.kind() != SOURCE_FILE && it.kind() != MODULE && it.kind() != CALL_EXPR
            })
            .find_map(ast::RecordExprField::cast);

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
            self.is_macro_call = path.syntax().parent().and_then(ast::MacroCall::cast).is_some();
            self.is_pattern_call =
                path.syntax().parent().and_then(ast::TupleStructPat::cast).is_some();

            self.is_path_type = path.syntax().parent().and_then(ast::PathType::cast).is_some();
            self.has_type_args = segment.generic_arg_list().is_some();

            if let Some(path) = path_or_use_tree_qualifier(&path) {
                self.path_qual = path
                    .segment()
                    .and_then(|it| {
                        find_node_with_range::<ast::PathSegment>(
                            original_file,
                            it.syntax().text_range(),
                        )
                    })
                    .map(|it| it.parent_path());
                return;
            }

            if let Some(segment) = path.segment() {
                if segment.coloncolon_token().is_some() {
                    return;
                }
            }

            self.is_trivial_path = true;

            // Find either enclosing expr statement (thing with `;`) or a
            // block. If block, check that we are the last expr.
            self.can_be_stmt = name_ref
                .syntax()
                .ancestors()
                .find_map(|node| {
                    if let Some(stmt) = ast::ExprStmt::cast(node.clone()) {
                        return Some(stmt.syntax().text_range() == name_ref.syntax().text_range());
                    }
                    if let Some(block) = ast::BlockExpr::cast(node) {
                        return Some(
                            block.tail_expr().map(|e| e.syntax().text_range())
                                == Some(name_ref.syntax().text_range()),
                        );
                    }
                    None
                })
                .unwrap_or(false);
            self.is_expr = path.syntax().parent().and_then(ast::PathExpr::cast).is_some();

            if let Some(off) = name_ref.syntax().text_range().start().checked_sub(2.into()) {
                if let Some(if_expr) =
                    self.sema.find_node_at_offset_with_macros::<ast::IfExpr>(original_file, off)
                {
                    if if_expr.syntax().text_range().end() < name_ref.syntax().text_range().start()
                    {
                        self.after_if = true;
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
                .and_then(|r| find_node_with_range(original_file, r));
            self.dot_receiver_is_ambiguous_float_literal =
                if let Some(ast::Expr::Literal(l)) = &self.dot_receiver {
                    match l.kind() {
                        ast::LiteralKind::FloatNumber { .. } => l.token().text().ends_with('.'),
                        _ => false,
                    }
                } else {
                    false
                };
        }
        if let Some(method_call_expr) = ast::MethodCallExpr::cast(parent) {
            // As above
            self.dot_receiver = method_call_expr
                .receiver()
                .map(|e| e.syntax().text_range())
                .and_then(|r| find_node_with_range(original_file, r));
            self.is_call = true;
        }
    }
}

fn find_node_with_range<N: AstNode>(syntax: &SyntaxNode, range: TextRange) -> Option<N> {
    syntax.covering_element(range).ancestors().find_map(N::cast)
}

fn is_node<N: AstNode>(node: &SyntaxNode) -> bool {
    match node.ancestors().find_map(N::cast) {
        None => false,
        Some(n) => n.syntax().text_range() == node.text_range(),
    }
}

fn path_or_use_tree_qualifier(path: &ast::Path) -> Option<ast::Path> {
    if let Some(qual) = path.qualifier() {
        return Some(qual);
    }
    let use_tree_list = path.syntax().ancestors().find_map(ast::UseTreeList::cast)?;
    let use_tree = use_tree_list.syntax().parent().and_then(ast::UseTree::cast)?;
    use_tree.path()
}
