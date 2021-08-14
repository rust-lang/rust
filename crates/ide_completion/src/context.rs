//! See `CompletionContext` structure.

use base_db::SourceDatabaseExt;
use hir::{Local, Name, ScopeDef, Semantics, SemanticsScope, Type, TypeInfo};
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

#[derive(Debug)]
pub(super) enum PathKind {
    Expr,
    Type,
}

#[derive(Debug)]
pub(crate) struct PathCompletionContext {
    /// If this is a call with () already there
    call_kind: Option<CallKind>,
    /// A single-indent path, like `foo`. `::foo` should not be considered a trivial path.
    pub(super) is_trivial_path: bool,
    /// If not a trivial path, the prefix (qualifier).
    pub(super) qualifier: Option<ast::Path>,
    /// Whether the qualifier comes from a use tree parent or not
    pub(super) use_tree_parent: bool,
    pub(super) kind: Option<PathKind>,
    /// Whether the path segment has type args or not.
    pub(super) has_type_args: bool,
    /// `true` if we are a statement or a last expr in the block.
    pub(super) can_be_stmt: bool,
    pub(super) in_loop_body: bool,
}

#[derive(Debug)]
pub(super) struct PatternContext {
    pub(super) refutability: PatternRefutability,
    pub(super) is_param: Option<ParamKind>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum CallKind {
    Pat,
    Mac,
    Expr,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ParamKind {
    Function,
    Closure,
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

    /// The parent function of the cursor position if it exists.
    pub(super) function_def: Option<ast::Fn>,
    /// The parent impl of the cursor position if it exists.
    pub(super) impl_def: Option<ast::Impl>,
    pub(super) name_ref_syntax: Option<ast::NameRef>,

    // potentially set if we are completing a lifetime
    pub(super) lifetime_syntax: Option<ast::Lifetime>,
    pub(super) lifetime_param_syntax: Option<ast::LifetimeParam>,
    pub(super) lifetime_allowed: bool,
    pub(super) is_label_ref: bool,

    pub(super) completion_location: Option<ImmediateLocation>,
    pub(super) prev_sibling: Option<ImmediatePrevSibling>,
    pub(super) attribute_under_caret: Option<ast::Attr>,
    pub(super) previous_token: Option<SyntaxToken>,

    pub(super) pattern_ctx: Option<PatternContext>,
    pub(super) path_context: Option<PathCompletionContext>,
    pub(super) active_parameter: Option<ActiveParameter>,
    pub(super) locals: Vec<(String, Local)>,

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
            expected_name: None,
            expected_type: None,
            function_def: None,
            impl_def: None,
            name_ref_syntax: None,
            lifetime_syntax: None,
            lifetime_param_syntax: None,
            lifetime_allowed: false,
            is_label_ref: false,
            pattern_ctx: None,
            completion_location: None,
            prev_sibling: None,
            attribute_under_caret: None,
            previous_token: None,
            path_context: None,
            active_parameter: ActiveParameter::at(db, position),
            locals,
            incomplete_let: false,
            no_completion_required: false,
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
        matches!(self.completion_location, Some(ImmediateLocation::Trait | ImmediateLocation::Impl))
    }

    pub(crate) fn has_dot_receiver(&self) -> bool {
        matches!(
            &self.completion_location,
            Some(ImmediateLocation::FieldAccess { receiver, .. } | ImmediateLocation::MethodCall { receiver,.. })
                if receiver.is_some()
        )
    }

    pub(crate) fn dot_receiver(&self) -> Option<&ast::Expr> {
        match &self.completion_location {
            Some(
                ImmediateLocation::MethodCall { receiver, .. }
                | ImmediateLocation::FieldAccess { receiver, .. },
            ) => receiver.as_ref(),
            _ => None,
        }
    }

    pub(crate) fn expects_non_trait_assoc_item(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::Impl))
    }

    pub(crate) fn expects_item(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::ItemList))
    }

    pub(crate) fn expects_generic_arg(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::GenericArgList(_)))
    }

    pub(crate) fn has_block_expr_parent(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::BlockExpr))
    }

    pub(crate) fn expects_ident_pat_or_ref_expr(&self) -> bool {
        matches!(
            self.completion_location,
            Some(ImmediateLocation::IdentPat | ImmediateLocation::RefExpr)
        )
    }

    pub(crate) fn expect_field(&self) -> bool {
        matches!(
            self.completion_location,
            Some(ImmediateLocation::RecordField | ImmediateLocation::TupleField)
        )
    }

    pub(crate) fn in_use_tree(&self) -> bool {
        matches!(
            self.completion_location,
            Some(ImmediateLocation::Use | ImmediateLocation::UseTree)
        )
    }

    pub(crate) fn has_impl_or_trait_prev_sibling(&self) -> bool {
        matches!(
            self.prev_sibling,
            Some(ImmediatePrevSibling::ImplDefType | ImmediatePrevSibling::TraitDefName)
        )
    }

    pub(crate) fn has_impl_prev_sibling(&self) -> bool {
        matches!(self.prev_sibling, Some(ImmediatePrevSibling::ImplDefType))
    }

    pub(crate) fn has_visibility_prev_sibling(&self) -> bool {
        matches!(self.prev_sibling, Some(ImmediatePrevSibling::Visibility))
    }

    pub(crate) fn after_if(&self) -> bool {
        matches!(self.prev_sibling, Some(ImmediatePrevSibling::IfExpr))
    }

    pub(crate) fn is_path_disallowed(&self) -> bool {
        self.attribute_under_caret.is_some()
            || self.previous_token_is(T![unsafe])
            || matches!(
                self.prev_sibling,
                Some(ImmediatePrevSibling::Attribute | ImmediatePrevSibling::Visibility)
            )
            || matches!(
                self.completion_location,
                Some(
                    ImmediateLocation::Attribute(_)
                        | ImmediateLocation::ModDeclaration(_)
                        | ImmediateLocation::RecordPat(_)
                        | ImmediateLocation::RecordExpr(_)
                )
            )
    }

    pub(crate) fn expects_expression(&self) -> bool {
        matches!(self.path_context, Some(PathCompletionContext { kind: Some(PathKind::Expr), .. }))
    }

    pub(crate) fn expects_type(&self) -> bool {
        matches!(self.path_context, Some(PathCompletionContext { kind: Some(PathKind::Type), .. }))
    }

    pub(crate) fn path_call_kind(&self) -> Option<CallKind> {
        self.path_context.as_ref().and_then(|it| it.call_kind)
    }

    pub(crate) fn is_trivial_path(&self) -> bool {
        matches!(self.path_context, Some(PathCompletionContext { is_trivial_path: true, .. }))
    }

    pub(crate) fn is_non_trivial_path(&self) -> bool {
        matches!(self.path_context, Some(PathCompletionContext { is_trivial_path: false, .. }))
    }

    pub(crate) fn path_qual(&self) -> Option<&ast::Path> {
        self.path_context.as_ref().and_then(|it| it.qualifier.as_ref())
    }

    /// Checks if an item is visible and not `doc(hidden)` at the completion site.
    pub(crate) fn is_visible<I>(&self, item: &I) -> bool
    where
        I: hir::HasVisibility + hir::HasAttrs + hir::HasCrate + Copy,
    {
        self.is_visible_impl(&item.visibility(self.db), &item.attrs(self.db), item.krate(self.db))
    }

    pub(crate) fn is_scope_def_hidden(&self, scope_def: &ScopeDef) -> bool {
        if let (Some(attrs), Some(krate)) = (scope_def.attrs(self.db), scope_def.krate(self.db)) {
            return self.is_doc_hidden(&attrs, krate);
        }

        false
    }

    pub(crate) fn is_item_hidden(&self, item: &hir::ItemInNs) -> bool {
        let attrs = item.attrs(self.db);
        let krate = item.krate(self.db);
        match (attrs, krate) {
            (Some(attrs), Some(krate)) => self.is_doc_hidden(&attrs, krate),
            _ => false,
        }
    }

    /// A version of [`SemanticsScope::process_all_names`] that filters out `#[doc(hidden)]` items.
    pub(crate) fn process_all_names(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        self.scope.process_all_names(&mut |name, def| {
            if self.is_scope_def_hidden(&def) {
                return;
            }

            f(name, def);
        })
    }

    fn is_visible_impl(
        &self,
        vis: &hir::Visibility,
        attrs: &hir::Attrs,
        defining_crate: hir::Crate,
    ) -> bool {
        let module = match self.scope.module() {
            Some(it) => it,
            None => return false,
        };
        if !vis.is_visible_from(self.db, module.into()) {
            // If the definition location is editable, also show private items
            let root_file = defining_crate.root_file(self.db);
            let source_root_id = self.db.file_source_root(root_file);
            let is_editable = !self.db.source_root(source_root_id).is_library;
            return is_editable;
        }

        !self.is_doc_hidden(attrs, defining_crate)
    }

    fn is_doc_hidden(&self, attrs: &hir::Attrs, defining_crate: hir::Crate) -> bool {
        let module = match self.scope.module() {
            Some(it) => it,
            None => return true,
        };
        if module.krate() != defining_crate && attrs.has_doc_hidden() {
            // `doc(hidden)` items are only completed within the defining crate.
            return true;
        }

        false
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
                            .or_else(|| it.initializer().and_then(|it| self.sema.type_of_expr(&it)))
                            .map(TypeInfo::original);
                        let name = if let Some(ast::Pat::IdentPat(ident)) = it.pat() {
                            ident.name().map(NameOrNameRef::Name)
                        } else {
                            None
                        };

                        (ty, name)
                    },
                    ast::ArgList(_it) => {
                        cov_mark::hit!(expected_type_fn_param);
                        ActiveParameter::at_token(
                            &self.sema,
                            self.token.clone(),
                        ).map(|ap| {
                            let name = ap.ident().map(NameOrNameRef::Name);
                            let ty = if has_ref(&self.token) {
                                cov_mark::hit!(expected_type_fn_param_ref);
                                ap.ty.remove_ref()
                            } else {
                                Some(ap.ty)
                            };
                            (ty, name)
                        })
                        .unwrap_or((None, None))
                    },
                    ast::RecordExprFieldList(it) => {
                        // wouldn't try {} be nice...
                        (|| {
                            if self.token.kind() == T![..]
                                || self.token.prev_token().map(|t| t.kind()) == Some(T![..])
                            {
                                cov_mark::hit!(expected_type_struct_func_update);
                                let record_expr = it.syntax().parent().and_then(ast::RecordExpr::cast)?;
                                let ty = self.sema.type_of_expr(&record_expr.into())?;
                                Some((
                                    Some(ty.original),
                                    None
                                ))
                            } else {
                                cov_mark::hit!(expected_type_struct_field_without_leading_char);
                                let expr_field = self.token.prev_sibling_or_token()?
                                    .into_node()
                                    .and_then(ast::RecordExprField::cast)?;
                                let (_, _, ty) = self.sema.resolve_record_field(&expr_field)?;
                                Some((
                                    Some(ty),
                                    expr_field.field_name().map(NameOrNameRef::NameRef),
                                ))
                            }
                        })().unwrap_or((None, None))
                    },
                    ast::RecordExprField(it) => {
                        cov_mark::hit!(expected_type_struct_field_with_leading_char);
                        (
                            it.expr().as_ref().and_then(|e| self.sema.type_of_expr(e)).map(TypeInfo::original),
                            it.field_name().map(NameOrNameRef::NameRef),
                        )
                    },
                    ast::MatchExpr(it) => {
                        cov_mark::hit!(expected_type_match_arm_without_leading_char);
                        let ty = it.expr().and_then(|e| self.sema.type_of_expr(&e)).map(TypeInfo::original);
                        (ty, None)
                    },
                    ast::IfExpr(it) => {
                        cov_mark::hit!(expected_type_if_let_without_leading_char);
                        let ty = it.condition()
                            .and_then(|cond| cond.expr())
                            .and_then(|e| self.sema.type_of_expr(&e))
                            .map(TypeInfo::original);
                        (ty, None)
                    },
                    ast::IdentPat(it) => {
                        cov_mark::hit!(expected_type_if_let_with_leading_char);
                        cov_mark::hit!(expected_type_match_arm_with_leading_char);
                        let ty = self.sema.type_of_pat(&ast::Pat::from(it)).map(TypeInfo::original);
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
                        ty.and_then(|ty| ty.original.as_callable(self.db))
                            .map(|c| (Some(c.return_type()), None))
                            .unwrap_or((None, None))
                    },
                    ast::Stmt(_it) => (None, None),
                    ast::Item(__) => (None, None),
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

        self.incomplete_let =
            syntax_element.ancestors().take(6).find_map(ast::LetStmt::cast).map_or(false, |it| {
                it.syntax().text_range().end() == syntax_element.text_range().end()
            });

        let (expected_type, expected_name) = self.expected_type_and_name();
        self.expected_type = expected_type;
        self.expected_name = expected_name;

        let name_like = match find_node_at_offset(&file_with_fake_ident, offset) {
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
        self.fill_impl_def();

        if let Some(bind_pat) = name.syntax().parent().and_then(ast::IdentPat::cast) {
            let is_name_in_field_pat = bind_pat
                .syntax()
                .parent()
                .and_then(ast::RecordPatField::cast)
                .map_or(false, |pat_field| pat_field.name_ref().is_none());
            if is_name_in_field_pat {
                return;
            }
            if bind_pat.is_simple_ident() {
                let mut is_param = None;
                let refutability = bind_pat
                    .syntax()
                    .ancestors()
                    .skip_while(|it| ast::Pat::can_cast(it.kind()))
                    .next()
                    .map_or(PatternRefutability::Irrefutable, |node| {
                        match_ast! {
                            match node {
                                ast::LetStmt(__) => PatternRefutability::Irrefutable,
                                ast::Param(param) => {
                                    let is_closure_param = param
                                        .syntax()
                                        .ancestors()
                                        .nth(2)
                                        .and_then(ast::ClosureExpr::cast)
                                        .is_some();
                                    is_param = Some(if is_closure_param {
                                        ParamKind::Closure
                                    } else {
                                        ParamKind::Function
                                    });
                                    PatternRefutability::Irrefutable
                                },
                                ast::MatchArm(__) => PatternRefutability::Refutable,
                                ast::Condition(__) => PatternRefutability::Refutable,
                                ast::ForExpr(__) => PatternRefutability::Irrefutable,
                                _ => PatternRefutability::Irrefutable,
                            }
                        }
                    });
                self.pattern_ctx = Some(PatternContext { refutability, is_param });
            }
        }
    }

    fn classify_name_ref(&mut self, original_file: &SyntaxNode, name_ref: ast::NameRef) {
        self.fill_impl_def();

        self.name_ref_syntax =
            find_node_at_offset(original_file, name_ref.syntax().text_range().start());

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
            let path_ctx = self.path_context.get_or_insert(PathCompletionContext {
                call_kind: None,
                is_trivial_path: false,
                qualifier: None,
                has_type_args: false,
                can_be_stmt: false,
                in_loop_body: false,
                use_tree_parent: false,
                kind: None,
            });
            path_ctx.in_loop_body = is_in_loop_body(name_ref.syntax());
            let path = segment.parent_path();

            if let Some(p) = path.syntax().parent() {
                path_ctx.call_kind = match_ast! {
                    match p {
                        ast::PathExpr(it) => it.syntax().parent().and_then(ast::CallExpr::cast).map(|_| CallKind::Expr),
                        ast::MacroCall(it) => it.excl_token().and(Some(CallKind::Mac)),
                        ast::TupleStructPat(_it) => Some(CallKind::Pat),
                        _ => None
                    }
                };
            }

            if let Some(parent) = path.syntax().parent() {
                path_ctx.kind = match_ast! {
                    match parent {
                        ast::PathType(_it) => Some(PathKind::Type),
                        ast::PathExpr(_it) => Some(PathKind::Expr),
                        _ => None,
                    }
                };
            }
            path_ctx.has_type_args = segment.generic_arg_list().is_some();

            if let Some((path, use_tree_parent)) = path_or_use_tree_qualifier(&path) {
                path_ctx.use_tree_parent = use_tree_parent;
                path_ctx.qualifier = path
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

            path_ctx.is_trivial_path = true;

            // Find either enclosing expr statement (thing with `;`) or a
            // block. If block, check that we are the last expr.
            path_ctx.can_be_stmt = name_ref
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
        }
    }
}

fn find_node_with_range<N: AstNode>(syntax: &SyntaxNode, range: TextRange) -> Option<N> {
    syntax.covering_element(range).ancestors().find_map(N::cast)
}

fn path_or_use_tree_qualifier(path: &ast::Path) -> Option<(ast::Path, bool)> {
    if let Some(qual) = path.qualifier() {
        return Some((qual, false));
    }
    let use_tree_list = path.syntax().ancestors().find_map(ast::UseTreeList::cast)?;
    let use_tree = use_tree_list.syntax().parent().and_then(ast::UseTree::cast)?;
    use_tree.path().zip(Some(true))
}

fn has_ref(token: &SyntaxToken) -> bool {
    let mut token = token.clone();
    for skip in [WHITESPACE, IDENT, T![mut]] {
        if token.kind() == skip {
            token = match token.prev_token() {
                Some(it) => it,
                None => return false,
            }
        }
    }
    token.kind() == T![&]
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use hir::HirDisplay;

    use crate::tests::{position, TEST_CONFIG};

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
    fn expected_type_fn_param() {
        cov_mark::check!(expected_type_fn_param);
        check_expected_type_and_name(
            r#"
fn foo() { bar($0); }
fn bar(x: u32) {}
"#,
            expect![[r#"ty: u32, name: x"#]],
        );
        check_expected_type_and_name(
            r#"
fn foo() { bar(c$0); }
fn bar(x: u32) {}
"#,
            expect![[r#"ty: u32, name: x"#]],
        );
    }

    #[test]
    fn expected_type_fn_param_ref() {
        cov_mark::check!(expected_type_fn_param_ref);
        check_expected_type_and_name(
            r#"
fn foo() { bar(&$0); }
fn bar(x: &u32) {}
"#,
            expect![[r#"ty: u32, name: x"#]],
        );
        check_expected_type_and_name(
            r#"
fn foo() { bar(&mut $0); }
fn bar(x: &mut u32) {}
"#,
            expect![[r#"ty: u32, name: x"#]],
        );
        check_expected_type_and_name(
            r#"
fn foo() { bar(&c$0); }
fn bar(x: &u32) {}
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
//- minicore: fn
fn foo() {
    bar(|| a$0);
}

fn bar(f: impl FnOnce() -> u32) {}
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

    #[test]
    fn expected_type_functional_update() {
        cov_mark::check!(expected_type_struct_func_update);
        check_expected_type_and_name(
            r#"
struct Foo { field: u32 }
fn foo() {
    Foo {
        ..$0
    }
}
"#,
            expect![[r#"ty: Foo, name: ?"#]],
        );
    }
}
