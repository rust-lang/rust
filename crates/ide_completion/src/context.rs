//! See `CompletionContext` structure.

use std::iter;

use base_db::SourceDatabaseExt;
use hir::{HasAttrs, Local, Name, ScopeDef, Semantics, SemanticsScope, Type, TypeInfo};
use ide_db::{
    active_parameter::ActiveParameter,
    base_db::{FilePosition, SourceDatabase},
    helpers::FamousDefs,
    RootDatabase,
};
use syntax::{
    algo::find_node_at_offset,
    ast::{self, HasName, NameOrNameRef},
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

#[derive(Copy, Clone, Debug)]
pub(super) enum PathKind {
    Expr,
    Type,
    Attr,
    Mac,
    Pat,
    Vis { has_in_token: bool },
    Use,
}

#[derive(Debug)]
pub(crate) struct PathCompletionContext {
    /// If this is a call with () already there
    has_call_parens: bool,
    /// A single-indent path, like `foo`. `::foo` should not be considered a trivial path.
    pub(super) is_trivial_path: bool,
    /// If not a trivial path, the prefix (qualifier).
    pub(super) qualifier: Option<ast::Path>,
    #[allow(dead_code)]
    /// If not a trivial path, the suffix (parent).
    pub(super) parent: Option<ast::Path>,
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
    pub(super) has_type_ascription: bool,
}

#[derive(Debug)]
pub(super) enum LifetimeContext {
    LifetimeParam(Option<ast::LifetimeParam>),
    Lifetime,
    LabelRef,
    LabelDef,
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
    /// The crate of the current file.
    pub(super) krate: Option<hir::Crate>,
    pub(super) expected_name: Option<NameOrNameRef>,
    pub(super) expected_type: Option<Type>,

    /// The parent function of the cursor position if it exists.
    pub(super) function_def: Option<ast::Fn>,
    pub(super) attr: Option<ast::Attr>,
    /// The parent impl of the cursor position if it exists.
    pub(super) impl_def: Option<ast::Impl>,
    /// The NameLike under the cursor in the original file if it exists.
    pub(super) name_syntax: Option<ast::NameLike>,
    pub(super) incomplete_let: bool,

    pub(super) completion_location: Option<ImmediateLocation>,
    pub(super) prev_sibling: Option<ImmediatePrevSibling>,
    pub(super) fake_attribute_under_caret: Option<ast::Attr>,
    pub(super) previous_token: Option<SyntaxToken>,

    pub(super) lifetime_ctx: Option<LifetimeContext>,
    pub(super) pattern_ctx: Option<PatternContext>,
    pub(super) path_context: Option<PathCompletionContext>,

    pub(super) locals: Vec<(Name, Local)>,

    no_completion_required: bool,
}

impl<'a> CompletionContext<'a> {
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

    pub(crate) fn famous_defs(&self) -> FamousDefs {
        FamousDefs(&self.sema, self.krate)
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

    pub(crate) fn has_dot_receiver(&self) -> bool {
        matches!(
            &self.completion_location,
            Some(ImmediateLocation::FieldAccess { receiver, .. } | ImmediateLocation::MethodCall { receiver,.. })
                if receiver.is_some()
        )
    }

    pub(crate) fn expects_assoc_item(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::Trait | ImmediateLocation::Impl))
    }

    pub(crate) fn expects_variant(&self) -> bool {
        matches!(self.completion_location, Some(ImmediateLocation::Variant))
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
        matches!(self.completion_location, Some(ImmediateLocation::StmtList))
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
        self.previous_token_is(T![unsafe])
            || matches!(
                self.prev_sibling,
                Some(ImmediatePrevSibling::Attribute | ImmediatePrevSibling::Visibility)
            )
            || matches!(
                self.completion_location,
                Some(
                    ImmediateLocation::ModDeclaration(_)
                        | ImmediateLocation::RecordPat(_)
                        | ImmediateLocation::RecordExpr(_)
                        | ImmediateLocation::Rename
                )
            )
    }

    pub(crate) fn expects_expression(&self) -> bool {
        matches!(self.path_context, Some(PathCompletionContext { kind: Some(PathKind::Expr), .. }))
    }

    pub(crate) fn expects_type(&self) -> bool {
        matches!(self.path_context, Some(PathCompletionContext { kind: Some(PathKind::Type), .. }))
    }

    pub(crate) fn path_is_call(&self) -> bool {
        self.path_context.as_ref().map_or(false, |it| it.has_call_parens)
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

    pub(crate) fn path_kind(&self) -> Option<PathKind> {
        self.path_context.as_ref().and_then(|it| it.kind)
    }

    /// Checks if an item is visible and not `doc(hidden)` at the completion site.
    pub(crate) fn is_visible<I>(&self, item: &I) -> bool
    where
        I: hir::HasVisibility + hir::HasAttrs + hir::HasCrate + Copy,
    {
        self.is_visible_impl(&item.visibility(self.db), &item.attrs(self.db), item.krate(self.db))
    }

    pub(crate) fn is_scope_def_hidden(&self, scope_def: ScopeDef) -> bool {
        if let (Some(attrs), Some(krate)) = (scope_def.attrs(self.db), scope_def.krate(self.db)) {
            return self.is_doc_hidden(&attrs, krate);
        }

        false
    }

    /// Check if an item is `#[doc(hidden)]`.
    pub(crate) fn is_item_hidden(&self, item: &hir::ItemInNs) -> bool {
        let attrs = item.attrs(self.db);
        let krate = item.krate(self.db);
        match (attrs, krate) {
            (Some(attrs), Some(krate)) => self.is_doc_hidden(&attrs, krate),
            _ => false,
        }
    }

    pub(crate) fn is_immediately_after_macro_bang(&self) -> bool {
        self.token.kind() == BANG && self.token.parent().map_or(false, |it| it.kind() == MACRO_CALL)
    }

    /// Whether the given trait is an operator trait or not.
    pub(crate) fn is_ops_trait(&self, trait_: hir::Trait) -> bool {
        match trait_.attrs(self.db).lang() {
            Some(lang) => OP_TRAIT_LANG_NAMES.contains(&lang.as_str()),
            None => false,
        }
    }

    /// A version of [`SemanticsScope::process_all_names`] that filters out `#[doc(hidden)]` items.
    pub(crate) fn process_all_names(&self, f: &mut dyn FnMut(Name, ScopeDef)) {
        let _p = profile::span("CompletionContext::process_all_names");
        self.scope.process_all_names(&mut |name, def| {
            if self.is_scope_def_hidden(def) {
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
        let krate = match self.krate {
            Some(it) => it,
            None => return true,
        };
        if krate != defining_crate && attrs.has_doc_hidden() {
            // `doc(hidden)` items are only completed within the defining crate.
            return true;
        }

        false
    }
}

// CompletionContext construction
impl<'a> CompletionContext<'a> {
    pub(super) fn new(
        db: &'a RootDatabase,
        position @ FilePosition { file_id, offset }: FilePosition,
        config: &'a CompletionConfig,
    ) -> Option<CompletionContext<'a>> {
        let _p = profile::span("CompletionContext::new");
        let sema = Semantics::new(db);

        let original_file = sema.parse(file_id);

        // Insert a fake ident to get a valid parse tree. We will use this file
        // to determine context, though the original_file will be used for
        // actual completion.
        let file_with_fake_ident = {
            let parse = db.parse(file_id);
            let edit = Indel::insert(offset, "intellijRulezz".to_string());
            parse.reparse(&edit).tree()
        };
        let fake_ident_token =
            file_with_fake_ident.syntax().token_at_offset(offset).right_biased()?;

        let original_token = original_file.syntax().token_at_offset(offset).left_biased()?;
        let token = sema.descend_into_macros_single(original_token.clone());
        let scope = sema.scope_at_offset(&token, offset);
        let krate = scope.krate();
        let mut locals = vec![];
        scope.process_all_names(&mut |name, scope| {
            if let ScopeDef::Local(local) = scope {
                locals.push((name, local));
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
            attr: None,
            impl_def: None,
            name_syntax: None,
            lifetime_ctx: None,
            pattern_ctx: None,
            completion_location: None,
            prev_sibling: None,
            fake_attribute_under_caret: None,
            previous_token: None,
            path_context: None,
            locals,
            incomplete_let: false,
            no_completion_required: false,
        };
        ctx.expand_and_fill(
            original_file.syntax().clone(),
            file_with_fake_ident.syntax().clone(),
            offset,
            fake_ident_token,
        );
        Some(ctx)
    }

    /// Do the attribute expansion at the current cursor position for both original file and fake file
    /// as long as possible. As soon as one of the two expansions fail we stop to stay in sync.
    fn expand_and_fill(
        &mut self,
        mut original_file: SyntaxNode,
        mut speculative_file: SyntaxNode,
        mut offset: TextSize,
        mut fake_ident_token: SyntaxToken,
    ) {
        let _p = profile::span("CompletionContext::expand_and_fill");
        'expansion: loop {
            let parent_item =
                |item: &ast::Item| item.syntax().ancestors().skip(1).find_map(ast::Item::cast);
            let ancestor_items = iter::successors(
                Option::zip(
                    find_node_at_offset::<ast::Item>(&original_file, offset),
                    find_node_at_offset::<ast::Item>(&speculative_file, offset),
                ),
                |(a, b)| parent_item(a).zip(parent_item(b)),
            );
            for (actual_item, item_with_fake_ident) in ancestor_items {
                match (
                    self.sema.expand_attr_macro(&actual_item),
                    self.sema.speculative_expand_attr_macro(
                        &actual_item,
                        &item_with_fake_ident,
                        fake_ident_token.clone(),
                    ),
                ) {
                    // maybe parent items have attributes
                    (None, None) => (),
                    // successful expansions
                    (Some(actual_expansion), Some((fake_expansion, fake_mapped_token))) => {
                        let new_offset = fake_mapped_token.text_range().start();
                        if new_offset > actual_expansion.text_range().end() {
                            break 'expansion;
                        }
                        original_file = actual_expansion;
                        speculative_file = fake_expansion;
                        fake_ident_token = fake_mapped_token;
                        offset = new_offset;
                        continue 'expansion;
                    }
                    // exactly one expansion failed, inconsistent state so stop expanding completely
                    _ => break 'expansion,
                }
            }

            // Expand fn-like macro calls
            if let (Some(actual_macro_call), Some(macro_call_with_fake_ident)) = (
                find_node_at_offset::<ast::MacroCall>(&original_file, offset),
                find_node_at_offset::<ast::MacroCall>(&speculative_file, offset),
            ) {
                let mac_call_path0 = actual_macro_call.path().as_ref().map(|s| s.syntax().text());
                let mac_call_path1 =
                    macro_call_with_fake_ident.path().as_ref().map(|s| s.syntax().text());
                if mac_call_path0 != mac_call_path1 {
                    break;
                }
                let speculative_args = match macro_call_with_fake_ident.token_tree() {
                    Some(tt) => tt,
                    None => break,
                };

                match (
                    self.sema.expand(&actual_macro_call),
                    self.sema.speculative_expand(
                        &actual_macro_call,
                        &speculative_args,
                        fake_ident_token.clone(),
                    ),
                ) {
                    // successful expansions
                    (Some(actual_expansion), Some((fake_expansion, fake_mapped_token))) => {
                        let new_offset = fake_mapped_token.text_range().start();
                        if new_offset > actual_expansion.text_range().end() {
                            break;
                        }
                        original_file = actual_expansion;
                        speculative_file = fake_expansion;
                        fake_ident_token = fake_mapped_token;
                        offset = new_offset;
                        continue;
                    }
                    _ => break,
                }
            }

            break;
        }

        self.fill(&original_file, speculative_file, offset);
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
                        let name = match it.pat() {
                            Some(ast::Pat::IdentPat(ident)) => ident.name().map(NameOrNameRef::Name),
                            Some(_) | None => None,
                        };

                        (ty, name)
                    },
                    ast::ArgList(_) => {
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
                        if let Some(expr) = it.expr() {
                            cov_mark::hit!(expected_type_struct_field_with_leading_char);
                            (
                                self.sema.type_of_expr(&expr).map(TypeInfo::original),
                                it.field_name().map(NameOrNameRef::NameRef),
                            )
                        } else {
                            cov_mark::hit!(expected_type_struct_field_followed_by_comma);
                            let ty = self.sema.resolve_record_field(&it)
                                .map(|(_, _, ty)| ty);
                            (
                                ty,
                                it.field_name().map(NameOrNameRef::NameRef),
                            )
                        }
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
                    ast::ParamList(_) => (None, None),
                    ast::Stmt(_) => (None, None),
                    ast::Item(_) => (None, None),
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
        self.no_completion_required = {
            let inside_impl_trait_block = inside_impl_trait_block(syntax_element.clone());
            let fn_is_prev = self.previous_token_is(T![fn]);
            let for_is_prev2 = for_is_prev2(syntax_element.clone());
            (fn_is_prev && !inside_impl_trait_block) || for_is_prev2
        };

        self.attr = self
            .sema
            .token_ancestors_with_macros(self.token.clone())
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::Attr::cast);
        self.fake_attribute_under_caret = syntax_element.ancestors().find_map(ast::Attr::cast);

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
        self.name_syntax =
            find_node_at_offset(original_file, name_like.syntax().text_range().start());
        self.impl_def = self
            .sema
            .token_ancestors_with_macros(self.token.clone())
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::Impl::cast);
        self.function_def = self
            .sema
            .token_ancestors_with_macros(self.token.clone())
            .take_while(|it| it.kind() != SOURCE_FILE && it.kind() != MODULE)
            .find_map(ast::Fn::cast);
        match name_like {
            ast::NameLike::Lifetime(lifetime) => {
                self.lifetime_ctx =
                    Self::classify_lifetime(&self.sema, original_file, lifetime, offset);
            }
            ast::NameLike::NameRef(name_ref) => {
                if let Some((path_ctx, pat_ctx)) =
                    Self::classify_name_ref(&self.sema, original_file, name_ref)
                {
                    self.path_context = Some(path_ctx);
                    self.pattern_ctx = pat_ctx;
                }
            }
            ast::NameLike::Name(name) => {
                self.pattern_ctx = Self::classify_name(&self.sema, name);
            }
        }
    }

    fn classify_lifetime(
        sema: &Semantics<RootDatabase>,
        original_file: &SyntaxNode,
        lifetime: ast::Lifetime,
        offset: TextSize,
    ) -> Option<LifetimeContext> {
        let parent = lifetime.syntax().parent()?;
        if parent.kind() == ERROR {
            return None;
        }

        Some(match_ast! {
            match parent {
                ast::LifetimeParam(_) => LifetimeContext::LifetimeParam(sema.find_node_at_offset_with_macros(original_file, offset)),
                ast::BreakExpr(_) => LifetimeContext::LabelRef,
                ast::ContinueExpr(_) => LifetimeContext::LabelRef,
                ast::Label(_) => LifetimeContext::LabelDef,
                _ => LifetimeContext::Lifetime,
            }
        })
    }

    fn classify_name(_sema: &Semantics<RootDatabase>, name: ast::Name) -> Option<PatternContext> {
        let bind_pat = name.syntax().parent().and_then(ast::IdentPat::cast)?;
        let is_name_in_field_pat = bind_pat
            .syntax()
            .parent()
            .and_then(ast::RecordPatField::cast)
            .map_or(false, |pat_field| pat_field.name_ref().is_none());
        if is_name_in_field_pat {
            return None;
        }
        if !bind_pat.is_simple_ident() {
            return None;
        }
        Some(pattern_context_for(bind_pat.into()))
    }

    fn classify_name_ref(
        _sema: &Semantics<RootDatabase>,
        original_file: &SyntaxNode,
        name_ref: ast::NameRef,
    ) -> Option<(PathCompletionContext, Option<PatternContext>)> {
        let parent = name_ref.syntax().parent()?;
        let segment = ast::PathSegment::cast(parent)?;
        let path = segment.parent_path();

        let mut path_ctx = PathCompletionContext {
            has_call_parens: false,
            is_trivial_path: false,
            qualifier: None,
            parent: None,
            has_type_args: false,
            can_be_stmt: false,
            in_loop_body: false,
            use_tree_parent: false,
            kind: None,
        };
        let mut pat_ctx = None;
        path_ctx.in_loop_body = is_in_loop_body(name_ref.syntax());

        path_ctx.kind  = path.syntax().ancestors().find_map(|it| {
            match_ast! {
                match it {
                    ast::PathType(_) => Some(PathKind::Type),
                    ast::PathExpr(it) => {
                        path_ctx.has_call_parens = it.syntax().parent().map_or(false, |it| ast::CallExpr::can_cast(it.kind()));
                        Some(PathKind::Expr)
                    },
                    ast::TupleStructPat(it) => {
                        path_ctx.has_call_parens = true;
                        pat_ctx = Some(pattern_context_for(it.into()));
                        Some(PathKind::Pat)
                    },
                    ast::RecordPat(it) => {
                        pat_ctx = Some(pattern_context_for(it.into()));
                        Some(PathKind::Pat)
                    },
                    ast::PathPat(it) => {
                        pat_ctx = Some(pattern_context_for(it.into()));
                        Some(PathKind::Pat)
                    },
                    ast::MacroCall(it) => it.excl_token().and(Some(PathKind::Mac)),
                    ast::Meta(_) => Some(PathKind::Attr),
                    ast::Visibility(it) => Some(PathKind::Vis { has_in_token: it.in_token().is_some() }),
                    ast::UseTree(_) => Some(PathKind::Use),
                    _ => None,
                }
            }
        });
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
            return Some((path_ctx, pat_ctx));
        }

        if let Some(segment) = path.segment() {
            if segment.coloncolon_token().is_some() {
                return Some((path_ctx, pat_ctx));
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
                if let Some(stmt_list) = ast::StmtList::cast(node) {
                    return Some(
                        stmt_list.tail_expr().map(|e| e.syntax().text_range())
                            == Some(name_ref.syntax().text_range()),
                    );
                }
                None
            })
            .unwrap_or(false);
        Some((path_ctx, pat_ctx))
    }
}

fn pattern_context_for(pat: ast::Pat) -> PatternContext {
    let mut is_param = None;
    let (refutability, has_type_ascription) =
    pat
        .syntax()
        .ancestors()
        .skip_while(|it| ast::Pat::can_cast(it.kind()))
        .next()
        .map_or((PatternRefutability::Irrefutable, false), |node| {
            let refutability = match_ast! {
                match node {
                    ast::LetStmt(let_) => return (PatternRefutability::Irrefutable, let_.ty().is_some()),
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
                        return (PatternRefutability::Irrefutable, param.ty().is_some())
                    },
                    ast::MatchArm(_) => PatternRefutability::Refutable,
                    ast::Condition(_) => PatternRefutability::Refutable,
                    ast::ForExpr(_) => PatternRefutability::Irrefutable,
                    _ => PatternRefutability::Irrefutable,
                }
            };
            (refutability, false)
        });
    PatternContext { refutability, is_param, has_type_ascription }
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
    for skip in [IDENT, WHITESPACE, T![mut]] {
        if token.kind() == skip {
            token = match token.prev_token() {
                Some(it) => it,
                None => return false,
            }
        }
    }
    token.kind() == T![&]
}

const OP_TRAIT_LANG_NAMES: &[&str] = &[
    "add_assign",
    "add",
    "bitand_assign",
    "bitand",
    "bitor_assign",
    "bitor",
    "bitxor_assign",
    "bitxor",
    "deref_mut",
    "deref",
    "div_assign",
    "div",
    "fn_mut",
    "fn_once",
    "fn",
    "index_mut",
    "index",
    "mul_assign",
    "mul",
    "neg",
    "not",
    "rem_assign",
    "rem",
    "shl_assign",
    "shl",
    "shr_assign",
    "shr",
    "sub",
    "sub",
];
#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use hir::HirDisplay;

    use crate::tests::{position, TEST_CONFIG};

    use super::CompletionContext;

    fn check_expected_type_and_name(ra_fixture: &str, expect: Expect) {
        let (db, pos) = position(ra_fixture);
        let config = TEST_CONFIG;
        let completion_context = CompletionContext::new(&db, pos, &config).unwrap();

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
fn foo() { bar(& c$0); }
fn bar(x: &u32) {}
        "#,
            expect![[r#"ty: u32, name: x"#]],
        );
        check_expected_type_and_name(
            r#"
fn foo() { bar(&mut c$0); }
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
    fn expected_type_struct_field_followed_by_comma() {
        cov_mark::check!(expected_type_struct_field_followed_by_comma);
        check_expected_type_and_name(
            r#"
struct Foo { a: u32 }
fn foo() {
    Foo { a: $0, };
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

    #[test]
    fn expected_type_param_pat() {
        check_expected_type_and_name(
            r#"
struct Foo { field: u32 }
fn foo(a$0: Foo) {}
"#,
            expect![[r#"ty: Foo, name: ?"#]],
        );
        check_expected_type_and_name(
            r#"
struct Foo { field: u32 }
fn foo($0: Foo) {}
"#,
            // FIXME make this work, currently fails due to pattern recovery eating the `:`
            expect![[r#"ty: ?, name: ?"#]],
        );
    }
}
