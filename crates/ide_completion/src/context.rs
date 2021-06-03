//! See `CompletionContext` structure.

use hir::{Local, ScopeDef, Semantics, SemanticsScope, Type};
use ide_db::{
    base_db::{FilePosition, SourceDatabase},
    call_info::ActiveParameter,
    RootDatabase,
};
use syntax::{
    algo::find_node_at_offset,
    ast::{self, NameOrNameRef, NameOwner},
    match_ast, AstNode, NodeOrToken,
    SyntaxKind::{self, *},
    SyntaxNode, SyntaxToken, TextRange, TextSize, T,
};
use text_edit::Indel;

use crate::{
    patterns::{
        determine_location, determine_prev_sibling, for_is_prev2, inside_impl_trait_block,
        is_in_loop_body, previous_token, ImmediateLocation, ImmediatePrevSibling,
    },
    CompletionConfig,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum PatternRefutability {
    Refutable,
    Irrefutable,
}

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
    pub(super) expected_name: Option<NameOrNameRef>,
    pub(super) expected_type: Option<Type>,
    pub(super) name_ref_syntax: Option<ast::NameRef>,

    pub(super) use_item_syntax: Option<ast::Use>,

    /// The parent function of the cursor position if it exists.
    pub(super) function_def: Option<ast::Fn>,
    /// The parent impl of the cursor position if it exists.
    pub(super) impl_def: Option<ast::Impl>,

    // potentially set if we are completing a lifetime
    pub(super) lifetime_syntax: Option<ast::Lifetime>,
    pub(super) lifetime_param_syntax: Option<ast::LifetimeParam>,
    pub(super) lifetime_allowed: bool,
    pub(super) is_label_ref: bool,

    // potentially set if we are completing a name
    pub(super) is_pat_or_const: Option<PatternRefutability>,
    pub(super) is_param: bool,

    pub(super) completion_location: Option<ImmediateLocation>,
    pub(super) prev_sibling: Option<ImmediatePrevSibling>,
    pub(super) attribute_under_caret: Option<ast::Attr>,

    /// FIXME: `ActiveParameter` is string-based, which is very very wrong
    pub(super) active_parameter: Option<ActiveParameter>,
    /// A single-indent path, like `foo`. `::foo` should not be considered a trivial path.
    pub(super) is_trivial_path: bool,
    /// If not a trivial path, the prefix (qualifier).
    pub(super) path_qual: Option<ast::Path>,
    /// `true` if we are a statement or a last expr in the block.
    pub(super) can_be_stmt: bool,
    /// `true` if we expect an expression at the cursor position.
    pub(super) is_expr: bool,
    /// If this is a call (method or function) in particular, i.e. the () are already there.
    pub(super) is_call: bool,
    /// Like `is_call`, but for tuple patterns.
    pub(super) is_pattern_call: bool,
    /// If this is a macro call, i.e. the () are already there.
    pub(super) is_macro_call: bool,
    pub(super) is_path_type: bool,
    pub(super) has_type_args: bool,
    pub(super) locals: Vec<(String, Local)>,

    pub(super) previous_token: Option<SyntaxToken>,
    pub(super) in_loop_body: bool,
    pub(super) incomplete_let: bool,

    no_completion_required: bool,
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
        let scope = sema.scope_at_offset(&token, position.offset);
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
            lifetime_allowed: false,
            expected_name: None,
            expected_type: None,
            name_ref_syntax: None,
            lifetime_syntax: None,
            lifetime_param_syntax: None,
            function_def: None,
            use_item_syntax: None,
            impl_def: None,
            active_parameter: ActiveParameter::at(db, position),
            is_label_ref: false,
            is_param: false,
            is_pat_or_const: None,
            is_trivial_path: false,
            path_qual: None,
            can_be_stmt: false,
            is_expr: false,
            is_call: false,
            is_pattern_call: false,
            is_macro_call: false,
            is_path_type: false,
            has_type_args: false,
            previous_token: None,
            in_loop_body: false,
            completion_location: None,
            prev_sibling: None,
            no_completion_required: false,
            incomplete_let: false,
            attribute_under_caret: None,
            locals,
        };

        let mut original_file = original_file.syntax().clone();
        let mut speculative_file = file_with_fake_ident.syntax().clone();
        let mut offset = position.offset;
        let mut fake_ident_token = fake_ident_token;

        // Are we inside a macro call?
        while let (Some(actual_macro_call), Some(macro_call_with_fake_ident)) = (
            find_node_at_offset::<ast::MacroCall>(&original_file, offset),
            find_node_at_offset::<ast::MacroCall>(&speculative_file, offset),
        ) {
            if actual_macro_call.path().as_ref().map(|s| s.syntax().text())
                != macro_call_with_fake_ident.path().as_ref().map(|s| s.syntax().text())
            {
                break;
            }
            let speculative_args = match macro_call_with_fake_ident.token_tree() {
                Some(tt) => tt,
                None => break,
            };
            if let (Some(actual_expansion), Some(speculative_expansion)) = (
                ctx.sema.expand(&actual_macro_call),
                ctx.sema.speculative_expand(
                    &actual_macro_call,
                    &speculative_args,
                    fake_ident_token,
                ),
            ) {
                let new_offset = speculative_expansion.1.text_range().start();
                if new_offset > actual_expansion.text_range().end() {
                    break;
                }
                original_file = actual_expansion;
                speculative_file = speculative_expansion.0;
                fake_ident_token = speculative_expansion.1;
                offset = new_offset;
            } else {
                break;
            }
        }
        ctx.fill(&original_file, speculative_file, offset);
        Some(ctx)
    }

    /// Checks whether completions in that particular case don't make much sense.
    /// Examples:
    /// - `fn $0` -- we expect function name, it's unlikely that "hint" will be helpful.
    ///   Exception for this case is `impl Trait for Foo`, where we would like to hint trait method names.
    /// - `for _ i$0` -- obviously, it'll be "in" keyword.
    pub(crate) fn no_completion_required(&self) -> bool {
        self.no_completion_required
    }

    /// The range of the identifier that is being completed.
    pub(crate) fn source_range(&self) -> TextRange {
        // check kind of macro-expanded token, but use range of original token
        let kind = self.token.kind();
        if kind == IDENT || kind == LIFETIME_IDENT || kind == UNDERSCORE || kind.is_keyword() {
            cov_mark::hit!(completes_if_prefix_is_keyword);
            self.original_token.text_range()
        } else if kind == CHAR {
            // assume we are completing a lifetime but the user has only typed the '
            cov_mark::hit!(completes_if_lifetime_without_idents);
            TextRange::at(self.original_token.text_range().start(), TextSize::from(1))
        } else {
            TextRange::empty(self.position.offset)
        }
    }

    pub(crate) fn previous_token_is(&self, kind: SyntaxKind) -> bool {
        self.previous_token.as_ref().map_or(false, |tok| tok.kind() == kind)
    }

    pub(crate) fn expects_assoc_item(&self) -> bool {
        matches!(
            self.completion_location,
            Some(ImmediateLocation::Trait) | Some(ImmediateLocation::Impl)
        )
    }

    pub(crate) fn has_dot_receiver(&self) -> bool {
        matches!(
            &self.completion_location,
            Some(ImmediateLocation::FieldAccess { receiver, .. }) | Some(ImmediateLocation::MethodCall { receiver })
                if receiver.is_some()
        )
    }

    pub(crate) fn dot_receiver(&self) -> Option<&ast::Expr> {
        match &self.completion_location {
            Some(ImmediateLocation::MethodCall { receiver })
            | Some(ImmediateLocation::FieldAccess { receiver, .. }) => receiver.as_ref(),
            _ => None,
        }
    }

    pub(crate) fn expects_use_tree(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::Use))
    }

    pub(crate) fn expects_non_trait_assoc_item(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::Impl))
    }

    pub(crate) fn expects_item(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::ItemList))
    }

    //         fn expects_value(&self) -> bool {
    pub(crate) fn expects_expression(&self) -> bool {
        self.is_expr
    }

    pub(crate) fn has_block_expr_parent(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::BlockExpr))
    }

    pub(crate) fn expects_ident_pat_or_ref_expr(&self) -> bool {
        matches!(
            self.completion_location,
            Some(ImmediateLocation::IdentPat) | Some(ImmediateLocation::RefExpr)
        )
    }

    pub(crate) fn expect_record_field(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::RecordField))
    }

    pub(crate) fn has_impl_or_trait_prev_sibling(&self) -> bool {
        matches!(
            self.prev_sibling,
            Some(ImmediatePrevSibling::ImplDefType) | Some(ImmediatePrevSibling::TraitDefName)
        )
    }

    pub(crate) fn after_if(&self) -> bool {
        matches!(self.prev_sibling, Some(ImmediatePrevSibling::IfExpr))
    }

    pub(crate) fn is_path_disallowed(&self) -> bool {
        matches!(
            self.completion_location,
            Some(ImmediateLocation::Attribute(_))
                | Some(ImmediateLocation::ModDeclaration(_))
                | Some(ImmediateLocation::RecordPat(_))
                | Some(ImmediateLocation::RecordExpr(_))
        ) || self.attribute_under_caret.is_some()
    }

    fn fill_impl_def(&mut self) {
        self.impl_def = self
            .sema
            .token_ancestors_with_macros(self.token.clone())
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::Impl::cast);
    }

    fn expected_type_and_name(&self) -> (Option<Type>, Option<NameOrNameRef>) {
        let mut node = match self.token.parent() {
            Some(it) => it,
            None => return (None, None),
        };
        loop {
            break match_ast! {
                match node {
                    ast::LetStmt(it) => {
                        cov_mark::hit!(expected_type_let_with_leading_char);
                        cov_mark::hit!(expected_type_let_without_leading_char);
                        let ty = it.pat()
                            .and_then(|pat| self.sema.type_of_pat(&pat))
                            .or_else(|| it.initializer().and_then(|it| self.sema.type_of_expr(&it)));
                        let name = if let Some(ast::Pat::IdentPat(ident)) = it.pat() {
                            ident.name().map(NameOrNameRef::Name)
                        } else {
                            None
                        };

                        (ty, name)
                    },
                    ast::ArgList(_it) => {
                        cov_mark::hit!(expected_type_fn_param_with_leading_char);
                        cov_mark::hit!(expected_type_fn_param_without_leading_char);
                        ActiveParameter::at_token(
                            &self.sema,
                            self.token.clone(),
                        ).map(|ap| {
                            let name = ap.ident().map(NameOrNameRef::Name);
                            (Some(ap.ty), name)
                        })
                        .unwrap_or((None, None))
                    },
                    ast::RecordExprFieldList(_it) => {
                        cov_mark::hit!(expected_type_struct_field_without_leading_char);
                        // wouldn't try {} be nice...
                        (|| {
                            let expr_field = self.token.prev_sibling_or_token()?
                                      .into_node()
                                      .and_then(|node| ast::RecordExprField::cast(node))?;
                            let (_, _, ty) = self.sema.resolve_record_field(&expr_field)?;
                            Some((
                                Some(ty),
                                expr_field.field_name().map(NameOrNameRef::NameRef),
                            ))
                        })().unwrap_or((None, None))
                    },
                    ast::RecordExprField(it) => {
                        cov_mark::hit!(expected_type_struct_field_with_leading_char);
                        (
                            it.expr().as_ref().and_then(|e| self.sema.type_of_expr(e)),
                            it.field_name().map(NameOrNameRef::NameRef),
                        )
                    },
                    ast::MatchExpr(it) => {
                        cov_mark::hit!(expected_type_match_arm_without_leading_char);
                        let ty = it.expr()
                            .and_then(|e| self.sema.type_of_expr(&e));
                        (ty, None)
                    },
                    ast::IfExpr(it) => {
                        cov_mark::hit!(expected_type_if_let_without_leading_char);
                        let ty = it.condition()
                            .and_then(|cond| cond.expr())
                            .and_then(|e| self.sema.type_of_expr(&e));
                        (ty, None)
                    },
                    ast::IdentPat(it) => {
                        cov_mark::hit!(expected_type_if_let_with_leading_char);
                        cov_mark::hit!(expected_type_match_arm_with_leading_char);
                        let ty = self.sema.type_of_pat(&ast::Pat::from(it));
                        (ty, None)
                    },
                    ast::Fn(it) => {
                        cov_mark::hit!(expected_type_fn_ret_with_leading_char);
                        cov_mark::hit!(expected_type_fn_ret_without_leading_char);
                        let def = self.sema.to_def(&it);
                        (def.map(|def| def.ret_type(self.db)), None)
                    },
                    ast::ClosureExpr(it) => {
                        let ty = self.sema.type_of_expr(&it.into());
                        ty.and_then(|ty| ty.as_callable(self.db))
                            .map(|c| (Some(c.return_type()), None))
                            .unwrap_or((None, None))
                    },
                    ast::Stmt(_it) => (None, None),
                    _ => {
                        match node.parent() {
                            Some(n) => {
                                node = n;
                                continue;
                            },
                            None => (None, None),
                        }
                    },
                }
            };
        }
    }

    fn fill(
        &mut self,
        original_file: &SyntaxNode,
        file_with_fake_ident: SyntaxNode,
        offset: TextSize,
    ) {
        let fake_ident_token = file_with_fake_ident.token_at_offset(offset).right_biased().unwrap();
        let syntax_element = NodeOrToken::Token(fake_ident_token);
        self.previous_token = previous_token(syntax_element.clone());
        self.attribute_under_caret = syntax_element.ancestors().find_map(ast::Attr::cast);
        self.no_completion_required = {
            let inside_impl_trait_block = inside_impl_trait_block(syntax_element.clone());
            let fn_is_prev = self.previous_token_is(T![fn]);
            let for_is_prev2 = for_is_prev2(syntax_element.clone());
            (fn_is_prev && !inside_impl_trait_block) || for_is_prev2
        };
        self.in_loop_body = is_in_loop_body(syntax_element.clone());

        self.incomplete_let =
            syntax_element.ancestors().take(6).find_map(ast::LetStmt::cast).map_or(false, |it| {
                it.syntax().text_range().end() == syntax_element.text_range().end()
            });

        let (expected_type, expected_name) = self.expected_type_and_name();
        self.expected_type = expected_type;
        self.expected_name = expected_name;

        let name_like = match find_node_at_offset(&&file_with_fake_ident, offset) {
            Some(it) => it,
            None => return,
        };
        self.completion_location =
            determine_location(&self.sema, original_file, offset, &name_like);
        self.prev_sibling = determine_prev_sibling(&name_like);
        match name_like {
            ast::NameLike::Lifetime(lifetime) => {
                self.classify_lifetime(original_file, lifetime, offset);
            }
            ast::NameLike::NameRef(name_ref) => {
                self.classify_name_ref(original_file, name_ref);
            }
            ast::NameLike::Name(name) => {
                self.classify_name(name);
            }
        }
    }

    fn classify_lifetime(
        &mut self,
        original_file: &SyntaxNode,
        lifetime: ast::Lifetime,
        offset: TextSize,
    ) {
        self.lifetime_syntax =
            find_node_at_offset(original_file, lifetime.syntax().text_range().start());
        if let Some(parent) = lifetime.syntax().parent() {
            if parent.kind() == ERROR {
                return;
            }

            match_ast! {
                match parent {
                    ast::LifetimeParam(_it) => {
                        self.lifetime_allowed = true;
                        self.lifetime_param_syntax =
                            self.sema.find_node_at_offset_with_macros(original_file, offset);
                    },
                    ast::BreakExpr(_it) => self.is_label_ref = true,
                    ast::ContinueExpr(_it) => self.is_label_ref = true,
                    ast::Label(_it) => (),
                    _ => self.lifetime_allowed = true,
                }
            }
        }
    }

    fn classify_name(&mut self, name: ast::Name) {
        if let Some(bind_pat) = name.syntax().parent().and_then(ast::IdentPat::cast) {
            self.is_pat_or_const = Some(PatternRefutability::Refutable);
            // if any of these is here our bind pat can't be a const pat anymore
            let complex_ident_pat = bind_pat.at_token().is_some()
                || bind_pat.ref_token().is_some()
                || bind_pat.mut_token().is_some();
            if complex_ident_pat {
                self.is_pat_or_const = None;
            } else {
                let irrefutable_pat = bind_pat.syntax().ancestors().find_map(|node| {
                    match_ast! {
                        match node {
                            ast::LetStmt(it) => Some(it.pat()),
                            ast::Param(it) => Some(it.pat()),
                            _ => None,
                        }
                    }
                });
                if let Some(Some(pat)) = irrefutable_pat {
                    // This check is here since we could be inside a pattern in the initializer expression of the let statement.
                    if pat.syntax().text_range().contains_range(bind_pat.syntax().text_range()) {
                        self.is_pat_or_const = Some(PatternRefutability::Irrefutable);
                    }
                }

                let is_name_in_field_pat = bind_pat
                    .syntax()
                    .parent()
                    .and_then(ast::RecordPatField::cast)
                    .map_or(false, |pat_field| pat_field.name_ref().is_none());
                if is_name_in_field_pat {
                    self.is_pat_or_const = None;
                }
            }

            self.fill_impl_def();
        }

        self.is_param |= is_node::<ast::Param>(name.syntax());
    }

    fn classify_name_ref(&mut self, original_file: &SyntaxNode, name_ref: ast::NameRef) {
        self.fill_impl_def();

        self.name_ref_syntax =
            find_node_at_offset(original_file, name_ref.syntax().text_range().start());

        if matches!(self.completion_location, Some(ImmediateLocation::ItemList)) {
            return;
        }

        self.use_item_syntax =
            self.sema.token_ancestors_with_macros(self.token.clone()).find_map(ast::Use::cast);

        self.function_def = self
            .sema
            .token_ancestors_with_macros(self.token.clone())
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::Fn::cast);

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
        }
        self.is_call |=
            matches!(self.completion_location, Some(ImmediateLocation::MethodCall { .. }));
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

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use hir::HirDisplay;

    use crate::test_utils::{position, TEST_CONFIG};

    use super::CompletionContext;

    fn check_expected_type_and_name(ra_fixture: &str, expect: Expect) {
        let (db, pos) = position(ra_fixture);
        let completion_context = CompletionContext::new(&db, pos, &TEST_CONFIG).unwrap();

        let ty = completion_context
            .expected_type
            .map(|t| t.display_test(&db).to_string())
            .unwrap_or("?".to_owned());

        let name = completion_context
            .expected_name
            .map_or_else(|| "?".to_owned(), |name| name.to_string());

        expect.assert_eq(&format!("ty: {}, name: {}", ty, name));
    }

    #[test]
    fn expected_type_let_without_leading_char() {
        cov_mark::check!(expected_type_let_without_leading_char);
        check_expected_type_and_name(
            r#"
fn foo() {
    let x: u32 = $0;
}
"#,
            expect![[r#"ty: u32, name: x"#]],
        );
    }

    #[test]
    fn expected_type_let_with_leading_char() {
        cov_mark::check!(expected_type_let_with_leading_char);
        check_expected_type_and_name(
            r#"
fn foo() {
    let x: u32 = c$0;
}
"#,
            expect![[r#"ty: u32, name: x"#]],
        );
    }

    #[test]
    fn expected_type_let_pat() {
        check_expected_type_and_name(
            r#"
fn foo() {
    let x$0 = 0u32;
}
"#,
            expect![[r#"ty: u32, name: ?"#]],
        );
        check_expected_type_and_name(
            r#"
fn foo() {
    let $0 = 0u32;
}
"#,
            expect![[r#"ty: u32, name: ?"#]],
        );
    }

    #[test]
    fn expected_type_fn_param_without_leading_char() {
        cov_mark::check!(expected_type_fn_param_without_leading_char);
        check_expected_type_and_name(
            r#"
fn foo() {
    bar($0);
}

fn bar(x: u32) {}
"#,
            expect![[r#"ty: u32, name: x"#]],
        );
    }

    #[test]
    fn expected_type_fn_param_with_leading_char() {
        cov_mark::check!(expected_type_fn_param_with_leading_char);
        check_expected_type_and_name(
            r#"
fn foo() {
    bar(c$0);
}

fn bar(x: u32) {}
"#,
            expect![[r#"ty: u32, name: x"#]],
        );
    }

    #[test]
    fn expected_type_struct_field_without_leading_char() {
        cov_mark::check!(expected_type_struct_field_without_leading_char);
        check_expected_type_and_name(
            r#"
struct Foo { a: u32 }
fn foo() {
    Foo { a: $0 };
}
"#,
            expect![[r#"ty: u32, name: a"#]],
        )
    }

    #[test]
    fn expected_type_generic_struct_field() {
        check_expected_type_and_name(
            r#"
struct Foo<T> { a: T }
fn foo() -> Foo<u32> {
    Foo { a: $0 }
}
"#,
            expect![[r#"ty: u32, name: a"#]],
        )
    }

    #[test]
    fn expected_type_struct_field_with_leading_char() {
        cov_mark::check!(expected_type_struct_field_with_leading_char);
        check_expected_type_and_name(
            r#"
struct Foo { a: u32 }
fn foo() {
    Foo { a: c$0 };
}
"#,
            expect![[r#"ty: u32, name: a"#]],
        );
    }

    #[test]
    fn expected_type_match_arm_without_leading_char() {
        cov_mark::check!(expected_type_match_arm_without_leading_char);
        check_expected_type_and_name(
            r#"
enum E { X }
fn foo() {
   match E::X { $0 }
}
"#,
            expect![[r#"ty: E, name: ?"#]],
        );
    }

    #[test]
    fn expected_type_match_arm_with_leading_char() {
        cov_mark::check!(expected_type_match_arm_with_leading_char);
        check_expected_type_and_name(
            r#"
enum E { X }
fn foo() {
   match E::X { c$0 }
}
"#,
            expect![[r#"ty: E, name: ?"#]],
        );
    }

    #[test]
    fn expected_type_if_let_without_leading_char() {
        cov_mark::check!(expected_type_if_let_without_leading_char);
        check_expected_type_and_name(
            r#"
enum Foo { Bar, Baz, Quux }

fn foo() {
    let f = Foo::Quux;
    if let $0 = f { }
}
"#,
            expect![[r#"ty: Foo, name: ?"#]],
        )
    }

    #[test]
    fn expected_type_if_let_with_leading_char() {
        cov_mark::check!(expected_type_if_let_with_leading_char);
        check_expected_type_and_name(
            r#"
enum Foo { Bar, Baz, Quux }

fn foo() {
    let f = Foo::Quux;
    if let c$0 = f { }
}
"#,
            expect![[r#"ty: Foo, name: ?"#]],
        )
    }

    #[test]
    fn expected_type_fn_ret_without_leading_char() {
        cov_mark::check!(expected_type_fn_ret_without_leading_char);
        check_expected_type_and_name(
            r#"
fn foo() -> u32 {
    $0
}
"#,
            expect![[r#"ty: u32, name: ?"#]],
        )
    }

    #[test]
    fn expected_type_fn_ret_with_leading_char() {
        cov_mark::check!(expected_type_fn_ret_with_leading_char);
        check_expected_type_and_name(
            r#"
fn foo() -> u32 {
    c$0
}
"#,
            expect![[r#"ty: u32, name: ?"#]],
        )
    }

    #[test]
    fn expected_type_fn_ret_fn_ref_fully_typed() {
        check_expected_type_and_name(
            r#"
fn foo() -> u32 {
    foo$0
}
"#,
            expect![[r#"ty: u32, name: ?"#]],
        )
    }

    #[test]
    fn expected_type_closure_param_return() {
        // FIXME: make this work with `|| $0`
        check_expected_type_and_name(
            r#"
fn foo() {
    bar(|| a$0);
}

fn bar(f: impl FnOnce() -> u32) {}
#[lang = "fn_once"]
trait FnOnce { type Output; }
"#,
            expect![[r#"ty: u32, name: ?"#]],
        );
    }

    #[test]
    fn expected_type_generic_function() {
        check_expected_type_and_name(
            r#"
fn foo() {
    bar::<u32>($0);
}

fn bar<T>(t: T) {}
"#,
            expect![[r#"ty: u32, name: t"#]],
        );
    }

    #[test]
    fn expected_type_generic_method() {
        check_expected_type_and_name(
            r#"
fn foo() {
    S(1u32).bar($0);
}

struct S<T>(T);
impl<T> S<T> {
    fn bar(self, t: T) {}
}
"#,
            expect![[r#"ty: u32, name: t"#]],
        );
    }
}
