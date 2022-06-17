//! Module responsible for analyzing the code surrounding the cursor for completion.
use std::iter;

use hir::{Semantics, Type, TypeInfo};
use ide_db::{active_parameter::ActiveParameter, RootDatabase};
use syntax::{
    algo::{find_node_at_offset, non_trivia_sibling},
    ast::{self, AttrKind, HasArgList, HasLoopBody, HasName, NameOrNameRef},
    match_ast, AstNode, AstToken, Direction, NodeOrToken, SyntaxElement, SyntaxKind, SyntaxNode,
    SyntaxToken, TextRange, TextSize, T,
};

use crate::context::{
    CompletionContext, DotAccess, DotAccessKind, IdentContext, ItemListKind, LifetimeContext,
    LifetimeKind, NameContext, NameKind, NameRefContext, NameRefKind, ParamKind, PathCompletionCtx,
    PathKind, PathQualifierCtx, PatternContext, PatternRefutability, QualifierCtx,
    TypeAscriptionTarget, TypeLocation, COMPLETION_MARKER,
};

impl<'a> CompletionContext<'a> {
    /// Expand attributes and macro calls at the current cursor position for both the original file
    /// and fake file repeatedly. As soon as one of the two expansions fail we stop so the original
    /// and speculative states stay in sync.
    pub(super) fn expand_and_fill(
        &mut self,
        mut original_file: SyntaxNode,
        mut speculative_file: SyntaxNode,
        mut offset: TextSize,
        mut fake_ident_token: SyntaxToken,
    ) -> Option<()> {
        let _p = profile::span("CompletionContext::expand_and_fill");
        let mut derive_ctx = None;

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

            // first try to expand attributes as these are always the outermost macro calls
            'ancestors: for (actual_item, item_with_fake_ident) in ancestor_items {
                match (
                    self.sema.expand_attr_macro(&actual_item),
                    self.sema.speculative_expand_attr_macro(
                        &actual_item,
                        &item_with_fake_ident,
                        fake_ident_token.clone(),
                    ),
                ) {
                    // maybe parent items have attributes, so continue walking the ancestors
                    (None, None) => continue 'ancestors,
                    // successful expansions
                    (Some(actual_expansion), Some((fake_expansion, fake_mapped_token))) => {
                        let new_offset = fake_mapped_token.text_range().start();
                        if new_offset > actual_expansion.text_range().end() {
                            // offset outside of bounds from the original expansion,
                            // stop here to prevent problems from happening
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

            // No attributes have been expanded, so look for macro_call! token trees or derive token trees
            let orig_tt = match find_node_at_offset::<ast::TokenTree>(&original_file, offset) {
                Some(it) => it,
                None => break 'expansion,
            };
            let spec_tt = match find_node_at_offset::<ast::TokenTree>(&speculative_file, offset) {
                Some(it) => it,
                None => break 'expansion,
            };

            // Expand pseudo-derive expansion
            if let (Some(orig_attr), Some(spec_attr)) = (
                orig_tt.syntax().parent().and_then(ast::Meta::cast).and_then(|it| it.parent_attr()),
                spec_tt.syntax().parent().and_then(ast::Meta::cast).and_then(|it| it.parent_attr()),
            ) {
                if let (Some(actual_expansion), Some((fake_expansion, fake_mapped_token))) = (
                    self.sema.expand_derive_as_pseudo_attr_macro(&orig_attr),
                    self.sema.speculative_expand_derive_as_pseudo_attr_macro(
                        &orig_attr,
                        &spec_attr,
                        fake_ident_token.clone(),
                    ),
                ) {
                    derive_ctx = Some((
                        actual_expansion,
                        fake_expansion,
                        fake_mapped_token.text_range().start(),
                        orig_attr,
                    ));
                }
                // at this point we won't have any more successful expansions, so stop
                break 'expansion;
            }

            // Expand fn-like macro calls
            if let (Some(actual_macro_call), Some(macro_call_with_fake_ident)) = (
                orig_tt.syntax().ancestors().find_map(ast::MacroCall::cast),
                spec_tt.syntax().ancestors().find_map(ast::MacroCall::cast),
            ) {
                let mac_call_path0 = actual_macro_call.path().as_ref().map(|s| s.syntax().text());
                let mac_call_path1 =
                    macro_call_with_fake_ident.path().as_ref().map(|s| s.syntax().text());

                // inconsistent state, stop expanding
                if mac_call_path0 != mac_call_path1 {
                    break 'expansion;
                }
                let speculative_args = match macro_call_with_fake_ident.token_tree() {
                    Some(tt) => tt,
                    None => break 'expansion,
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
                            // offset outside of bounds from the original expansion,
                            // stop here to prevent problems from happening
                            break 'expansion;
                        }
                        original_file = actual_expansion;
                        speculative_file = fake_expansion;
                        fake_ident_token = fake_mapped_token;
                        offset = new_offset;
                        continue 'expansion;
                    }
                    // at least on expansion failed, we won't have anything to expand from this point
                    // onwards so break out
                    _ => break 'expansion,
                }
            }

            // none of our states have changed so stop the loop
            break 'expansion;
        }

        self.fill(&original_file, speculative_file, offset, derive_ctx)
    }

    /// Calculate the expected type and name of the cursor position.
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
                    ast::LetExpr(it) => {
                        cov_mark::hit!(expected_type_if_let_without_leading_char);
                        let ty = it.pat()
                            .and_then(|pat| self.sema.type_of_pat(&pat))
                            .or_else(|| it.expr().and_then(|it| self.sema.type_of_expr(&it)))
                            .map(TypeInfo::original);
                        (ty, None)
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
                    // match foo { $0 }
                    // match foo { ..., pat => $0 }
                    ast::MatchExpr(it) => {
                        let ty = if self.previous_token_is(T![=>]) {
                            // match foo { ..., pat => $0 }
                            cov_mark::hit!(expected_type_match_arm_body_without_leading_char);
                            cov_mark::hit!(expected_type_match_arm_body_with_leading_char);
                            self.sema.type_of_expr(&it.into())
                        } else {
                            // match foo { $0 }
                            cov_mark::hit!(expected_type_match_arm_without_leading_char);
                            it.expr().and_then(|e| self.sema.type_of_expr(&e))
                        }.map(TypeInfo::original);
                        (ty, None)
                    },
                    ast::IfExpr(it) => {
                        let ty = it.condition()
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

    /// Fill the completion context, this is what does semantic reasoning about the surrounding context
    /// of the completion location.
    fn fill(
        &mut self,
        original_file: &SyntaxNode,
        file_with_fake_ident: SyntaxNode,
        offset: TextSize,
        derive_ctx: Option<(SyntaxNode, SyntaxNode, TextSize, ast::Attr)>,
    ) -> Option<()> {
        let fake_ident_token = file_with_fake_ident.token_at_offset(offset).right_biased()?;
        let syntax_element = NodeOrToken::Token(fake_ident_token);
        if is_in_token_of_for_loop(syntax_element.clone()) {
            // for pat $0
            // there is nothing to complete here except `in` keyword
            // don't bother populating the context
            // FIXME: the completion calculations should end up good enough
            // such that this special case becomes unnecessary
            return None;
        }

        self.previous_token = previous_token(syntax_element.clone());

        self.incomplete_let =
            syntax_element.ancestors().take(6).find_map(ast::LetStmt::cast).map_or(false, |it| {
                it.syntax().text_range().end() == syntax_element.text_range().end()
            });

        (self.expected_type, self.expected_name) = self.expected_type_and_name();

        // Overwrite the path kind for derives
        if let Some((original_file, file_with_fake_ident, offset, origin_attr)) = derive_ctx {
            self.existing_derives = self
                .sema
                .resolve_derive_macro(&origin_attr)
                .into_iter()
                .flatten()
                .flatten()
                .collect();

            if let Some(ast::NameLike::NameRef(name_ref)) =
                find_node_at_offset(&file_with_fake_ident, offset)
            {
                let parent = name_ref.syntax().parent()?;
                let (mut nameref_ctx, _, _) =
                    Self::classify_name_ref(&self.sema, &original_file, name_ref, parent);
                if let Some(NameRefKind::Path(path_ctx)) = &mut nameref_ctx.kind {
                    path_ctx.kind = PathKind::Derive;
                }
                self.ident_ctx = IdentContext::NameRef(nameref_ctx);
                return Some(());
            }
            return None;
        }

        let name_like = match find_node_at_offset(&file_with_fake_ident, offset) {
            Some(it) => it,
            None => {
                if let Some(original) = ast::String::cast(self.original_token.clone()) {
                    self.ident_ctx = IdentContext::String {
                        original,
                        expanded: ast::String::cast(self.token.clone()),
                    };
                } else {
                    // Fix up trailing whitespace problem
                    // #[attr(foo = $0
                    let token = if self.token.kind() == SyntaxKind::WHITESPACE {
                        self.previous_token.as_ref()?
                    } else {
                        &self.token
                    };
                    let p = token.parent()?;
                    if p.kind() == SyntaxKind::TOKEN_TREE
                        && p.ancestors().any(|it| it.kind() == SyntaxKind::META)
                    {
                        self.ident_ctx = IdentContext::UnexpandedAttrTT {
                            fake_attribute_under_caret: syntax_element
                                .ancestors()
                                .find_map(ast::Attr::cast),
                        };
                    } else {
                        return None;
                    }
                }
                return Some(());
            }
        };
        self.impl_def = self
            .sema
            .token_ancestors_with_macros(self.token.clone())
            .take_while(|it| it.kind() != SyntaxKind::SOURCE_FILE)
            .filter_map(ast::Item::cast)
            .take(2)
            .find_map(|it| match it {
                ast::Item::Impl(impl_) => Some(impl_),
                _ => None,
            });
        self.function_def = self
            .sema
            .token_ancestors_with_macros(self.token.clone())
            .take_while(|it| {
                it.kind() != SyntaxKind::SOURCE_FILE && it.kind() != SyntaxKind::MODULE
            })
            .filter_map(ast::Item::cast)
            .take(2)
            .find_map(|it| match it {
                ast::Item::Fn(fn_) => Some(fn_),
                _ => None,
            });

        match name_like {
            ast::NameLike::Lifetime(lifetime) => {
                self.ident_ctx = IdentContext::Lifetime(Self::classify_lifetime(
                    &self.sema,
                    original_file,
                    lifetime,
                )?);
            }
            ast::NameLike::NameRef(name_ref) => {
                let parent = name_ref.syntax().parent()?;
                let (nameref_ctx, pat_ctx, qualifier_ctx) =
                    Self::classify_name_ref(&self.sema, &original_file, name_ref, parent.clone());

                self.qualifier_ctx = qualifier_ctx;
                self.ident_ctx = IdentContext::NameRef(nameref_ctx);
                self.pattern_ctx = pat_ctx;
            }
            ast::NameLike::Name(name) => {
                let (name_ctx, pat_ctx) = Self::classify_name(&self.sema, original_file, name)?;
                self.pattern_ctx = pat_ctx;
                self.ident_ctx = IdentContext::Name(name_ctx);
            }
        }
        Some(())
    }

    fn classify_lifetime(
        _sema: &Semantics<RootDatabase>,
        original_file: &SyntaxNode,
        lifetime: ast::Lifetime,
    ) -> Option<LifetimeContext> {
        let parent = lifetime.syntax().parent()?;
        if parent.kind() == SyntaxKind::ERROR {
            return None;
        }

        let kind = match_ast! {
            match parent {
                ast::LifetimeParam(param) => LifetimeKind::LifetimeParam {
                    is_decl: param.lifetime().as_ref() == Some(&lifetime),
                    param
                },
                ast::BreakExpr(_) => LifetimeKind::LabelRef,
                ast::ContinueExpr(_) => LifetimeKind::LabelRef,
                ast::Label(_) => LifetimeKind::LabelDef,
                _ => LifetimeKind::Lifetime,
            }
        };
        let lifetime = find_node_at_offset(&original_file, lifetime.syntax().text_range().start());

        Some(LifetimeContext { lifetime, kind })
    }

    fn classify_name(
        _sema: &Semantics<RootDatabase>,
        original_file: &SyntaxNode,
        name: ast::Name,
    ) -> Option<(NameContext, Option<PatternContext>)> {
        let parent = name.syntax().parent()?;
        let mut pat_ctx = None;
        let kind = match_ast! {
            match parent {
                ast::Const(_) => NameKind::Const,
                ast::ConstParam(_) => NameKind::ConstParam,
                ast::Enum(_) => NameKind::Enum,
                ast::Fn(_) => NameKind::Function,
                ast::IdentPat(bind_pat) => {
                    pat_ctx = Some({
                        let mut pat_ctx = pattern_context_for(original_file, bind_pat.into());
                        if let Some(record_field) = ast::RecordPatField::for_field_name(&name) {
                            pat_ctx.record_pat = find_node_in_file_compensated(original_file, &record_field.parent_record_pat());
                        }
                        pat_ctx
                    });

                    NameKind::IdentPat
                },
                ast::MacroDef(_) => NameKind::MacroDef,
                ast::MacroRules(_) => NameKind::MacroRules,
                ast::Module(module) => NameKind::Module(module),
                ast::RecordField(_) => NameKind::RecordField,
                ast::Rename(_) => NameKind::Rename,
                ast::SelfParam(_) => NameKind::SelfParam,
                ast::Static(_) => NameKind::Static,
                ast::Struct(_) => NameKind::Struct,
                ast::Trait(_) => NameKind::Trait,
                ast::TypeAlias(_) => NameKind::TypeAlias,
                ast::TypeParam(_) => NameKind::TypeParam,
                ast::Union(_) => NameKind::Union,
                ast::Variant(_) => NameKind::Variant,
                _ => return None,
            }
        };
        let name = find_node_at_offset(&original_file, name.syntax().text_range().start());
        Some((NameContext { name, kind }, pat_ctx))
    }

    fn classify_name_ref(
        sema: &Semantics<RootDatabase>,
        original_file: &SyntaxNode,
        name_ref: ast::NameRef,
        parent: SyntaxNode,
    ) -> (NameRefContext, Option<PatternContext>, QualifierCtx) {
        let nameref = find_node_at_offset(&original_file, name_ref.syntax().text_range().start());

        let mut res = (NameRefContext { nameref, kind: None }, None, QualifierCtx::default());
        let (nameref_ctx, pattern_ctx, qualifier_ctx) = &mut res;

        if let Some(record_field) = ast::RecordExprField::for_field_name(&name_ref) {
            nameref_ctx.kind =
                find_node_in_file_compensated(original_file, &record_field.parent_record_lit())
                    .map(NameRefKind::RecordExpr);
            return res;
        }
        if let Some(record_field) = ast::RecordPatField::for_field_name_ref(&name_ref) {
            *pattern_ctx = Some(PatternContext {
                param_ctx: None,
                has_type_ascription: false,
                ref_token: None,
                mut_token: None,
                record_pat: find_node_in_file_compensated(
                    original_file,
                    &record_field.parent_record_pat(),
                ),
                ..pattern_context_for(
                    original_file,
                    record_field.parent_record_pat().clone().into(),
                )
            });
            return res;
        }

        let segment = match_ast! {
            match parent {
                ast::PathSegment(segment) => segment,
                ast::FieldExpr(field) => {
                    let receiver = find_opt_node_in_file(original_file, field.expr());
                    let receiver_is_ambiguous_float_literal = match &receiver {
                        Some(ast::Expr::Literal(l)) => matches! {
                            l.kind(),
                            ast::LiteralKind::FloatNumber { .. } if l.syntax().last_token().map_or(false, |it| it.text().ends_with('.'))
                        },
                        _ => false,
                    };
                    nameref_ctx.kind = Some(NameRefKind::DotAccess(DotAccess {
                        receiver_ty: receiver.as_ref().and_then(|it| sema.type_of_expr(it)),
                        kind: DotAccessKind::Field { receiver_is_ambiguous_float_literal },
                        receiver
                    }));
                    return res;
                },
                ast::MethodCallExpr(method) => {
                    let receiver = find_opt_node_in_file(original_file, method.receiver());
                    nameref_ctx.kind = Some(NameRefKind::DotAccess(DotAccess {
                        receiver_ty: receiver.as_ref().and_then(|it| sema.type_of_expr(it)),
                        kind: DotAccessKind::Method { has_parens: method.arg_list().map_or(false, |it| it.l_paren_token().is_some()) },
                        receiver
                    }));
                    return res;
                },
                _ => return res,
            }
        };

        let path = segment.parent_path();
        let mut path_ctx = PathCompletionCtx {
            has_call_parens: false,
            has_macro_bang: false,
            is_absolute_path: false,
            qualifier: None,
            parent: path.parent_path(),
            kind: PathKind::Item { kind: ItemListKind::SourceFile },
            has_type_args: false,
        };

        let is_in_block = |it: &SyntaxNode| {
            it.parent()
                .map(|node| {
                    ast::ExprStmt::can_cast(node.kind()) || ast::StmtList::can_cast(node.kind())
                })
                .unwrap_or(false)
        };
        let func_update_record = |syn: &SyntaxNode| {
            if let Some(record_expr) = syn.ancestors().nth(2).and_then(ast::RecordExpr::cast) {
                find_node_in_file_compensated(original_file, &record_expr)
            } else {
                None
            }
        };
        let after_if_expr = |node: SyntaxNode| {
            let prev_expr = (|| {
                let prev_sibling = non_trivia_sibling(node.into(), Direction::Prev)?.into_node()?;
                ast::ExprStmt::cast(prev_sibling)?.expr()
            })();
            matches!(prev_expr, Some(ast::Expr::IfExpr(_)))
        };

        // We do not want to generate path completions when we are sandwiched between an item decl signature and its body.
        // ex. trait Foo $0 {}
        // in these cases parser recovery usually kicks in for our inserted identifier, causing it
        // to either be parsed as an ExprStmt or a MacroCall, depending on whether it is in a block
        // expression or an item list.
        // The following code checks if the body is missing, if it is we either cut off the body
        // from the item or it was missing in the first place
        let inbetween_body_and_decl_check = |node: SyntaxNode| {
            if let Some(NodeOrToken::Node(n)) =
                syntax::algo::non_trivia_sibling(node.into(), syntax::Direction::Prev)
            {
                if let Some(item) = ast::Item::cast(n) {
                    let is_inbetween = match &item {
                        ast::Item::Const(it) => it.body().is_none(),
                        ast::Item::Enum(it) => it.variant_list().is_none(),
                        ast::Item::ExternBlock(it) => it.extern_item_list().is_none(),
                        ast::Item::Fn(it) => it.body().is_none(),
                        ast::Item::Impl(it) => it.assoc_item_list().is_none(),
                        ast::Item::Module(it) => it.item_list().is_none(),
                        ast::Item::Static(it) => it.body().is_none(),
                        ast::Item::Struct(it) => it.field_list().is_none(),
                        ast::Item::Trait(it) => it.assoc_item_list().is_none(),
                        ast::Item::TypeAlias(it) => it.ty().is_none(),
                        ast::Item::Union(it) => it.record_field_list().is_none(),
                        _ => false,
                    };
                    if is_inbetween {
                        return Some(item);
                    }
                }
            }
            None
        };

        let type_location = |it: Option<SyntaxNode>| {
            let parent = it?;
            let res = match_ast! {
                match parent {
                    ast::Const(it) => {
                        let name = find_opt_node_in_file(original_file, it.name())?;
                        let original = ast::Const::cast(name.syntax().parent()?)?;
                        TypeLocation::TypeAscription(TypeAscriptionTarget::Const(original.body()))
                    },
                    ast::RetType(it) => {
                        if it.thin_arrow_token().is_none() {
                            return None;
                        }
                        let parent = match ast::Fn::cast(parent.parent()?) {
                            Some(x) => x.param_list(),
                            None => ast::ClosureExpr::cast(parent.parent()?)?.param_list(),
                        };

                        let parent = find_opt_node_in_file(original_file, parent)?.syntax().parent()?;
                        TypeLocation::TypeAscription(TypeAscriptionTarget::RetType(match_ast! {
                            match parent {
                                ast::ClosureExpr(it) => {
                                    it.body()
                                },
                                ast::Fn(it) => {
                                    it.body().map(ast::Expr::BlockExpr)
                                },
                                _ => return None,
                            }
                        }))
                    },
                    ast::Param(it) => {
                        if it.colon_token().is_none() {
                            return None;
                        }
                        TypeLocation::TypeAscription(TypeAscriptionTarget::FnParam(find_opt_node_in_file(original_file, it.pat())))
                    },
                    ast::LetStmt(it) => {
                        if it.colon_token().is_none() {
                            return None;
                        }
                        TypeLocation::TypeAscription(TypeAscriptionTarget::Let(find_opt_node_in_file(original_file, it.pat())))
                    },
                    ast::TypeBound(_) => TypeLocation::TypeBound,
                    // is this case needed?
                    ast::TypeBoundList(_) => TypeLocation::TypeBound,
                    ast::GenericArg(it) => TypeLocation::GenericArgList(find_opt_node_in_file_compensated(original_file, it.syntax().parent().and_then(ast::GenericArgList::cast))),
                    // is this case needed?
                    ast::GenericArgList(it) => TypeLocation::GenericArgList(find_opt_node_in_file_compensated(original_file, Some(it))),
                    ast::TupleField(_) => TypeLocation::TupleField,
                    _ => return None,
                }
            };
            Some(res)
        };

        // Infer the path kind
        let kind = path.syntax().parent().and_then(|it| {
            match_ast! {
                match it {
                    ast::PathType(it) => {
                        let location = type_location(it.syntax().parent());
                        Some(PathKind::Type {
                            location: location.unwrap_or(TypeLocation::Other),
                        })
                    },
                    ast::PathExpr(it) => {
                        if let Some(p) = it.syntax().parent() {
                            if ast::ExprStmt::can_cast(p.kind()) {
                                if let Some(kind) = inbetween_body_and_decl_check(p) {
                                    nameref_ctx.kind = Some(NameRefKind::Keyword(kind));
                                    return None;
                                }
                            }
                        }

                        path_ctx.has_call_parens = it.syntax().parent().map_or(false, |it| ast::CallExpr::can_cast(it.kind()));
                        let in_block_expr = is_in_block(it.syntax());
                        let in_loop_body = is_in_loop_body(it.syntax());
                        let after_if_expr = after_if_expr(it.syntax().clone());
                        let ref_expr_parent = path.as_single_name_ref()
                            .and_then(|_| it.syntax().parent()).and_then(ast::RefExpr::cast);
                        let is_func_update = func_update_record(it.syntax());

                        Some(PathKind::Expr { in_block_expr, in_loop_body, after_if_expr, ref_expr_parent, is_func_update })
                    },
                    ast::TupleStructPat(it) => {
                        path_ctx.has_call_parens = true;
                        *pattern_ctx = Some(pattern_context_for(original_file, it.into()));
                        Some(PathKind::Pat)
                    },
                    ast::RecordPat(it) => {
                        path_ctx.has_call_parens = true;
                        *pattern_ctx = Some(pattern_context_for(original_file, it.into()));
                        Some(PathKind::Pat)
                    },
                    ast::PathPat(it) => {
                        *pattern_ctx = Some(pattern_context_for(original_file, it.into()));
                        Some(PathKind::Pat)
                    },
                    ast::MacroCall(it) => {
                        if let Some(kind) = inbetween_body_and_decl_check(it.syntax().clone()) {
                            nameref_ctx.kind = Some(NameRefKind::Keyword(kind));
                            return None;
                        }

                        path_ctx.has_macro_bang = it.excl_token().is_some();
                        let parent = it.syntax().parent();
                        match parent.as_ref().map(|it| it.kind()) {
                            Some(SyntaxKind::MACRO_PAT) => Some(PathKind::Pat),
                            Some(SyntaxKind::MACRO_TYPE) => {
                                let location = type_location(parent.unwrap().parent());
                                Some(PathKind::Type {
                                    location: location.unwrap_or(TypeLocation::Other),
                                })
                            },
                            Some(SyntaxKind::ITEM_LIST) => Some(PathKind::Item { kind: ItemListKind::Module }),
                            Some(SyntaxKind::ASSOC_ITEM_LIST) => Some(PathKind::Item { kind: match parent.and_then(|it| it.parent()) {
                                Some(it) => match_ast! {
                                    match it {
                                        ast::Trait(_) => ItemListKind::Trait,
                                        ast::Impl(it) => if it.trait_().is_some() {
                                            ItemListKind::TraitImpl
                                        } else {
                                            ItemListKind::Impl
                                        },
                                        _ => return None
                                    }
                                },
                                None => return None,
                            } }),
                            Some(SyntaxKind::EXTERN_ITEM_LIST) => Some(PathKind::Item { kind: ItemListKind::ExternBlock }),
                            Some(SyntaxKind::SOURCE_FILE) => Some(PathKind::Item { kind: ItemListKind::SourceFile }),
                            _ => {
                               return parent.and_then(ast::MacroExpr::cast).map(|it| {
                                    let in_loop_body = is_in_loop_body(it.syntax());
                                    let in_block_expr = is_in_block(it.syntax());
                                    let after_if_expr = after_if_expr(it.syntax().clone());
                                    let ref_expr_parent = path.as_single_name_ref()
                                        .and_then(|_| it.syntax().parent()).and_then(ast::RefExpr::cast);
                                    let is_func_update = func_update_record(it.syntax());
                                    PathKind::Expr { in_block_expr, in_loop_body, after_if_expr, ref_expr_parent, is_func_update }
                                });
                            },
                        }
                    },
                    ast::Meta(meta) => (|| {
                        let attr = meta.parent_attr()?;
                        let kind = attr.kind();
                        let attached = attr.syntax().parent()?;
                        let is_trailing_outer_attr = kind != AttrKind::Inner
                            && non_trivia_sibling(attr.syntax().clone().into(), syntax::Direction::Next).is_none();
                        let annotated_item_kind = if is_trailing_outer_attr {
                            None
                        } else {
                            Some(attached.kind())
                        };
                        Some(PathKind::Attr {
                            kind,
                            annotated_item_kind,
                        })
                    })(),
                    ast::Visibility(it) => Some(PathKind::Vis { has_in_token: it.in_token().is_some() }),
                    ast::UseTree(_) => Some(PathKind::Use),
                    _ => return None,
                }
            }
        });

        match kind {
            Some(kind) => path_ctx.kind = kind,
            None => return res,
        }
        path_ctx.has_type_args = segment.generic_arg_list().is_some();

        if let Some((path, use_tree_parent)) = path_or_use_tree_qualifier(&path) {
            if !use_tree_parent {
                path_ctx.is_absolute_path =
                    path.top_path().segment().map_or(false, |it| it.coloncolon_token().is_some());
            }

            let path = path
                .segment()
                .and_then(|it| find_node_in_file(original_file, &it))
                .map(|it| it.parent_path());
            path_ctx.qualifier = path.map(|path| {
                let res = sema.resolve_path(&path);
                let is_super_chain = iter::successors(Some(path.clone()), |p| p.qualifier())
                    .all(|p| p.segment().and_then(|s| s.super_token()).is_some());

                // `<_>::$0`
                let is_infer_qualifier = path.qualifier().is_none()
                    && matches!(
                        path.segment().and_then(|it| it.kind()),
                        Some(ast::PathSegmentKind::Type {
                            type_ref: Some(ast::Type::InferType(_)),
                            trait_ref: None,
                        })
                    );

                PathQualifierCtx {
                    path,
                    resolution: res,
                    is_super_chain,
                    use_tree_parent,
                    is_infer_qualifier,
                }
            });
        } else if let Some(segment) = path.segment() {
            if segment.coloncolon_token().is_some() {
                path_ctx.is_absolute_path = true;
            }
        }

        if path_ctx.is_trivial_path() {
            // fetch the full expression that may have qualifiers attached to it
            let top_node = match path_ctx.kind {
                PathKind::Expr { in_block_expr: true, .. } => {
                    parent.ancestors().find(|it| ast::PathExpr::can_cast(it.kind())).and_then(|p| {
                        let parent = p.parent()?;
                        if ast::StmtList::can_cast(parent.kind()) {
                            Some(p)
                        } else if ast::ExprStmt::can_cast(parent.kind()) {
                            Some(parent)
                        } else {
                            None
                        }
                    })
                }
                PathKind::Item { .. } => {
                    parent.ancestors().find(|it| ast::MacroCall::can_cast(it.kind()))
                }
                _ => None,
            };
            if let Some(top) = top_node {
                if let Some(NodeOrToken::Node(error_node)) =
                    syntax::algo::non_trivia_sibling(top.clone().into(), syntax::Direction::Prev)
                {
                    if error_node.kind() == SyntaxKind::ERROR {
                        qualifier_ctx.unsafe_tok = error_node
                            .children_with_tokens()
                            .filter_map(NodeOrToken::into_token)
                            .find(|it| it.kind() == T![unsafe]);
                        qualifier_ctx.vis_node =
                            error_node.children().find_map(ast::Visibility::cast);
                    }
                }

                if let PathKind::Item { .. } = path_ctx.kind {
                    if qualifier_ctx.none() {
                        if let Some(t) = top.first_token() {
                            if let Some(prev) = t
                                .prev_token()
                                .and_then(|t| syntax::algo::skip_trivia_token(t, Direction::Prev))
                            {
                                if ![T![;], T!['}'], T!['{']].contains(&prev.kind()) {
                                    // This was inferred to be an item position path, but it seems
                                    // to be part of some other broken node which leaked into an item
                                    // list, so return without setting the path context
                                    return res;
                                }
                            }
                        }
                    }
                }
            }
        }
        nameref_ctx.kind = Some(NameRefKind::Path(path_ctx));
        res
    }
}

fn pattern_context_for(original_file: &SyntaxNode, pat: ast::Pat) -> PatternContext {
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
                        let has_type_ascription = param.ty().is_some();
                        is_param = (|| {
                            let fake_param_list = param.syntax().parent().and_then(ast::ParamList::cast)?;
                            let param_list = find_node_in_file_compensated(original_file, &fake_param_list)?;
                            let param_list_owner = param_list.syntax().parent()?;
                            let kind = match_ast! {
                                match param_list_owner {
                                    ast::ClosureExpr(closure) => ParamKind::Closure(closure),
                                    ast::Fn(fn_) => ParamKind::Function(fn_),
                                    _ => return None,
                                }
                            };
                            Some((param_list, param, kind))
                        })();
                        return (PatternRefutability::Irrefutable, has_type_ascription)
                    },
                    ast::MatchArm(_) => PatternRefutability::Refutable,
                    ast::LetExpr(_) => PatternRefutability::Refutable,
                    ast::ForExpr(_) => PatternRefutability::Irrefutable,
                    _ => PatternRefutability::Irrefutable,
                }
            };
            (refutability, false)
        });
    let (ref_token, mut_token) = match &pat {
        ast::Pat::IdentPat(it) => (it.ref_token(), it.mut_token()),
        _ => (None, None),
    };
    PatternContext {
        refutability,
        param_ctx: is_param,
        has_type_ascription,
        parent_pat: pat.syntax().parent().and_then(ast::Pat::cast),
        mut_token,
        ref_token,
        record_pat: None,
    }
}

/// Attempts to find `node` inside `syntax` via `node`'s text range.
/// If the fake identifier has been inserted after this node or inside of this node use the `_compensated` version instead.
fn find_opt_node_in_file<N: AstNode>(syntax: &SyntaxNode, node: Option<N>) -> Option<N> {
    find_node_in_file(syntax, &node?)
}

/// Attempts to find `node` inside `syntax` via `node`'s text range.
/// If the fake identifier has been inserted after this node or inside of this node use the `_compensated` version instead.
fn find_node_in_file<N: AstNode>(syntax: &SyntaxNode, node: &N) -> Option<N> {
    let syntax_range = syntax.text_range();
    let range = node.syntax().text_range();
    let intersection = range.intersect(syntax_range)?;
    syntax.covering_element(intersection).ancestors().find_map(N::cast)
}

/// Attempts to find `node` inside `syntax` via `node`'s text range while compensating
/// for the offset introduced by the fake ident.
/// This is wrong if `node` comes before the insertion point! Use `find_node_in_file` instead.
fn find_node_in_file_compensated<N: AstNode>(syntax: &SyntaxNode, node: &N) -> Option<N> {
    let syntax_range = syntax.text_range();
    let range = node.syntax().text_range();
    let end = range.end().checked_sub(TextSize::try_from(COMPLETION_MARKER.len()).ok()?)?;
    if end < range.start() {
        return None;
    }
    let range = TextRange::new(range.start(), end);
    // our inserted ident could cause `range` to go outside of the original syntax, so cap it
    let intersection = range.intersect(syntax_range)?;
    syntax.covering_element(intersection).ancestors().find_map(N::cast)
}

/// Attempts to find `node` inside `syntax` via `node`'s text range while compensating
/// for the offset introduced by the fake ident..
/// This is wrong if `node` comes before the insertion point! Use `find_node_in_file` instead.
fn find_opt_node_in_file_compensated<N: AstNode>(
    syntax: &SyntaxNode,
    node: Option<N>,
) -> Option<N> {
    find_node_in_file_compensated(syntax, &node?)
}

fn path_or_use_tree_qualifier(path: &ast::Path) -> Option<(ast::Path, bool)> {
    if let Some(qual) = path.qualifier() {
        return Some((qual, false));
    }
    let use_tree_list = path.syntax().ancestors().find_map(ast::UseTreeList::cast)?;
    let use_tree = use_tree_list.syntax().parent().and_then(ast::UseTree::cast)?;
    Some((use_tree.path()?, true))
}

fn has_ref(token: &SyntaxToken) -> bool {
    let mut token = token.clone();
    for skip in [SyntaxKind::IDENT, SyntaxKind::WHITESPACE, T![mut]] {
        if token.kind() == skip {
            token = match token.prev_token() {
                Some(it) => it,
                None => return false,
            }
        }
    }
    token.kind() == T![&]
}

pub(crate) fn previous_token(element: SyntaxElement) -> Option<SyntaxToken> {
    element.into_token().and_then(previous_non_trivia_token)
}

pub(crate) fn is_in_token_of_for_loop(element: SyntaxElement) -> bool {
    // oh my ...
    (|| {
        let syntax_token = element.into_token()?;
        let range = syntax_token.text_range();
        let for_expr = syntax_token.parent_ancestors().find_map(ast::ForExpr::cast)?;

        // check if the current token is the `in` token of a for loop
        if let Some(token) = for_expr.in_token() {
            return Some(syntax_token == token);
        }
        let pat = for_expr.pat()?;
        if range.end() < pat.syntax().text_range().end() {
            // if we are inside or before the pattern we can't be at the `in` token position
            return None;
        }
        let next_sibl = next_non_trivia_sibling(pat.syntax().clone().into())?;
        Some(match next_sibl {
            // the loop body is some node, if our token is at the start we are at the `in` position,
            // otherwise we could be in a recovered expression, we don't wanna ruin completions there
            syntax::NodeOrToken::Node(n) => n.text_range().start() == range.start(),
            // the loop body consists of a single token, if we are this we are certainly at the `in` token position
            syntax::NodeOrToken::Token(t) => t == syntax_token,
        })
    })()
    .unwrap_or(false)
}

#[test]
fn test_for_is_prev2() {
    crate::tests::check_pattern_is_applicable(r"fn __() { for i i$0 }", is_in_token_of_for_loop);
}

pub(crate) fn is_in_loop_body(node: &SyntaxNode) -> bool {
    node.ancestors()
        .take_while(|it| it.kind() != SyntaxKind::FN && it.kind() != SyntaxKind::CLOSURE_EXPR)
        .find_map(|it| {
            let loop_body = match_ast! {
                match it {
                    ast::ForExpr(it) => it.loop_body(),
                    ast::WhileExpr(it) => it.loop_body(),
                    ast::LoopExpr(it) => it.loop_body(),
                    _ => None,
                }
            };
            loop_body.filter(|it| it.syntax().text_range().contains_range(node.text_range()))
        })
        .is_some()
}

fn previous_non_trivia_token(token: SyntaxToken) -> Option<SyntaxToken> {
    let mut token = token.prev_token();
    while let Some(inner) = token {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            token = inner.prev_token();
        }
    }
    None
}

fn next_non_trivia_sibling(ele: SyntaxElement) -> Option<SyntaxElement> {
    let mut e = ele.next_sibling_or_token();
    while let Some(inner) = e {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            e = inner.next_sibling_or_token();
        }
    }
    None
}
