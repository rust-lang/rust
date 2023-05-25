//! Module responsible for analyzing the code surrounding the cursor for completion.
use std::iter;

use hir::{Semantics, Type, TypeInfo, Variant};
use ide_db::{active_parameter::ActiveParameter, RootDatabase};
use syntax::{
    algo::{find_node_at_offset, non_trivia_sibling},
    ast::{self, AttrKind, HasArgList, HasLoopBody, HasName, NameOrNameRef},
    match_ast, AstNode, AstToken, Direction, NodeOrToken, SyntaxElement, SyntaxKind, SyntaxNode,
    SyntaxToken, TextRange, TextSize, T,
};

use crate::context::{
    AttrCtx, CompletionAnalysis, DotAccess, DotAccessKind, ExprCtx, ItemListKind, LifetimeContext,
    LifetimeKind, NameContext, NameKind, NameRefContext, NameRefKind, ParamContext, ParamKind,
    PathCompletionCtx, PathKind, PatternContext, PatternRefutability, Qualified, QualifierCtx,
    TypeAscriptionTarget, TypeLocation, COMPLETION_MARKER,
};

struct ExpansionResult {
    original_file: SyntaxNode,
    speculative_file: SyntaxNode,
    offset: TextSize,
    fake_ident_token: SyntaxToken,
    derive_ctx: Option<(SyntaxNode, SyntaxNode, TextSize, ast::Attr)>,
}

pub(super) struct AnalysisResult {
    pub(super) analysis: CompletionAnalysis,
    pub(super) expected: (Option<Type>, Option<ast::NameOrNameRef>),
    pub(super) qualifier_ctx: QualifierCtx,
    /// the original token of the expanded file
    pub(super) token: SyntaxToken,
    pub(super) offset: TextSize,
}

pub(super) fn expand_and_analyze(
    sema: &Semantics<'_, RootDatabase>,
    original_file: SyntaxNode,
    speculative_file: SyntaxNode,
    offset: TextSize,
    original_token: &SyntaxToken,
) -> Option<AnalysisResult> {
    // as we insert after the offset, right biased will *always* pick the identifier no matter
    // if there is an ident already typed or not
    let fake_ident_token = speculative_file.token_at_offset(offset).right_biased()?;
    // the relative offset between the cursor and the *identifier* token we are completing on
    let relative_offset = offset - fake_ident_token.text_range().start();
    // make the offset point to the start of the original token, as that is what the
    // intermediate offsets calculated in expansion always points to
    let offset = offset - relative_offset;
    let expansion =
        expand(sema, original_file, speculative_file, offset, fake_ident_token, relative_offset);

    // add the relative offset back, so that left_biased finds the proper token
    let offset = expansion.offset + relative_offset;
    let token = expansion.original_file.token_at_offset(offset).left_biased()?;

    analyze(sema, expansion, original_token, &token).map(|(analysis, expected, qualifier_ctx)| {
        AnalysisResult { analysis, expected, qualifier_ctx, token, offset }
    })
}

/// Expand attributes and macro calls at the current cursor position for both the original file
/// and fake file repeatedly. As soon as one of the two expansions fail we stop so the original
/// and speculative states stay in sync.
fn expand(
    sema: &Semantics<'_, RootDatabase>,
    mut original_file: SyntaxNode,
    mut speculative_file: SyntaxNode,
    mut offset: TextSize,
    mut fake_ident_token: SyntaxToken,
    relative_offset: TextSize,
) -> ExpansionResult {
    let _p = profile::span("CompletionContext::expand");
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
                sema.expand_attr_macro(&actual_item),
                sema.speculative_expand_attr_macro(
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
                    if new_offset + relative_offset > actual_expansion.text_range().end() {
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
                sema.expand_derive_as_pseudo_attr_macro(&orig_attr),
                sema.speculative_expand_derive_as_pseudo_attr_macro(
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
                sema.expand(&actual_macro_call),
                sema.speculative_expand(
                    &actual_macro_call,
                    &speculative_args,
                    fake_ident_token.clone(),
                ),
            ) {
                // successful expansions
                (Some(actual_expansion), Some((fake_expansion, fake_mapped_token))) => {
                    let new_offset = fake_mapped_token.text_range().start();
                    if new_offset + relative_offset > actual_expansion.text_range().end() {
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
    ExpansionResult { original_file, speculative_file, offset, fake_ident_token, derive_ctx }
}

/// Fill the completion context, this is what does semantic reasoning about the surrounding context
/// of the completion location.
fn analyze(
    sema: &Semantics<'_, RootDatabase>,
    expansion_result: ExpansionResult,
    original_token: &SyntaxToken,
    self_token: &SyntaxToken,
) -> Option<(CompletionAnalysis, (Option<Type>, Option<ast::NameOrNameRef>), QualifierCtx)> {
    let _p = profile::span("CompletionContext::analyze");
    let ExpansionResult { original_file, speculative_file, offset, fake_ident_token, derive_ctx } =
        expansion_result;

    // Overwrite the path kind for derives
    if let Some((original_file, file_with_fake_ident, offset, origin_attr)) = derive_ctx {
        if let Some(ast::NameLike::NameRef(name_ref)) =
            find_node_at_offset(&file_with_fake_ident, offset)
        {
            let parent = name_ref.syntax().parent()?;
            let (mut nameref_ctx, _) = classify_name_ref(sema, &original_file, name_ref, parent)?;
            if let NameRefKind::Path(path_ctx) = &mut nameref_ctx.kind {
                path_ctx.kind = PathKind::Derive {
                    existing_derives: sema
                        .resolve_derive_macro(&origin_attr)
                        .into_iter()
                        .flatten()
                        .flatten()
                        .collect(),
                };
            }
            return Some((
                CompletionAnalysis::NameRef(nameref_ctx),
                (None, None),
                QualifierCtx::default(),
            ));
        }
        return None;
    }

    let Some(name_like) = find_node_at_offset(&speculative_file, offset) else {
        let analysis = if let Some(original) = ast::String::cast(original_token.clone()) {
            CompletionAnalysis::String {
                original,
                expanded: ast::String::cast(self_token.clone()),
            }
        } else {
            // Fix up trailing whitespace problem
            // #[attr(foo = $0
            let token = syntax::algo::skip_trivia_token(self_token.clone(), Direction::Prev)?;
            let p = token.parent()?;
            if p.kind() == SyntaxKind::TOKEN_TREE
                && p.ancestors().any(|it| it.kind() == SyntaxKind::META)
            {
                let colon_prefix = previous_non_trivia_token(self_token.clone())
                    .map_or(false, |it| T![:] == it.kind());
                CompletionAnalysis::UnexpandedAttrTT {
                    fake_attribute_under_caret: fake_ident_token
                        .parent_ancestors()
                        .find_map(ast::Attr::cast),
                    colon_prefix,
                }
            } else {
                return None;
            }
        };
        return Some((analysis, (None, None), QualifierCtx::default()));
    };

    let expected = expected_type_and_name(sema, self_token, &name_like);
    let mut qual_ctx = QualifierCtx::default();
    let analysis = match name_like {
        ast::NameLike::Lifetime(lifetime) => {
            CompletionAnalysis::Lifetime(classify_lifetime(sema, &original_file, lifetime)?)
        }
        ast::NameLike::NameRef(name_ref) => {
            let parent = name_ref.syntax().parent()?;
            let (nameref_ctx, qualifier_ctx) =
                classify_name_ref(sema, &original_file, name_ref, parent)?;

            if let NameRefContext {
                kind:
                    NameRefKind::Path(PathCompletionCtx { kind: PathKind::Expr { .. }, path, .. }, ..),
                ..
            } = &nameref_ctx
            {
                if is_in_token_of_for_loop(path) {
                    // for pat $0
                    // there is nothing to complete here except `in` keyword
                    // don't bother populating the context
                    // Ideally this special casing wouldn't be needed, but the parser recovers
                    return None;
                }
            }

            qual_ctx = qualifier_ctx;
            CompletionAnalysis::NameRef(nameref_ctx)
        }
        ast::NameLike::Name(name) => {
            let name_ctx = classify_name(sema, &original_file, name)?;
            CompletionAnalysis::Name(name_ctx)
        }
    };
    Some((analysis, expected, qual_ctx))
}

/// Calculate the expected type and name of the cursor position.
fn expected_type_and_name(
    sema: &Semantics<'_, RootDatabase>,
    token: &SyntaxToken,
    name_like: &ast::NameLike,
) -> (Option<Type>, Option<NameOrNameRef>) {
    let mut node = match token.parent() {
        Some(it) => it,
        None => return (None, None),
    };

    let strip_refs = |mut ty: Type| match name_like {
        ast::NameLike::NameRef(n) => {
            let p = match n.syntax().parent() {
                Some(it) => it,
                None => return ty,
            };
            let top_syn = match_ast! {
                match p {
                    ast::FieldExpr(e) => e
                        .syntax()
                        .ancestors()
                        .take_while(|it| ast::FieldExpr::can_cast(it.kind()))
                        .last(),
                    ast::PathSegment(e) => e
                        .syntax()
                        .ancestors()
                        .skip(1)
                        .take_while(|it| ast::Path::can_cast(it.kind()) || ast::PathExpr::can_cast(it.kind()))
                        .find(|it| ast::PathExpr::can_cast(it.kind())),
                    _ => None
                }
            };
            let top_syn = match top_syn {
                Some(it) => it,
                None => return ty,
            };
            for _ in top_syn.ancestors().skip(1).map_while(ast::RefExpr::cast) {
                cov_mark::hit!(expected_type_fn_param_ref);
                ty = ty.strip_reference();
            }
            ty
        }
        _ => ty,
    };

    let (ty, name) = loop {
        break match_ast! {
            match node {
                ast::LetStmt(it) => {
                    cov_mark::hit!(expected_type_let_with_leading_char);
                    cov_mark::hit!(expected_type_let_without_leading_char);
                    let ty = it.pat()
                        .and_then(|pat| sema.type_of_pat(&pat))
                        .or_else(|| it.initializer().and_then(|it| sema.type_of_expr(&it)))
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
                        .and_then(|pat| sema.type_of_pat(&pat))
                        .or_else(|| it.expr().and_then(|it| sema.type_of_expr(&it)))
                        .map(TypeInfo::original);
                    (ty, None)
                },
                ast::ArgList(_) => {
                    cov_mark::hit!(expected_type_fn_param);
                    ActiveParameter::at_token(
                        sema,
                       token.clone(),
                    ).map(|ap| {
                        let name = ap.ident().map(NameOrNameRef::Name);
                        (Some(ap.ty), name)
                    })
                    .unwrap_or((None, None))
                },
                ast::RecordExprFieldList(it) => {
                    // wouldn't try {} be nice...
                    (|| {
                        if token.kind() == T![..]
                            ||token.prev_token().map(|t| t.kind()) == Some(T![..])
                        {
                            cov_mark::hit!(expected_type_struct_func_update);
                            let record_expr = it.syntax().parent().and_then(ast::RecordExpr::cast)?;
                            let ty = sema.type_of_expr(&record_expr.into())?;
                            Some((
                                Some(ty.original),
                                None
                            ))
                        } else {
                            cov_mark::hit!(expected_type_struct_field_without_leading_char);
                            let expr_field = token.prev_sibling_or_token()?
                                .into_node()
                                .and_then(ast::RecordExprField::cast)?;
                            let (_, _, ty) = sema.resolve_record_field(&expr_field)?;
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
                            sema.type_of_expr(&expr).map(TypeInfo::original),
                            it.field_name().map(NameOrNameRef::NameRef),
                        )
                    } else {
                        cov_mark::hit!(expected_type_struct_field_followed_by_comma);
                        let ty = sema.resolve_record_field(&it)
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
                    let on_arrow = previous_non_trivia_token(token.clone()).map_or(false, |it| T![=>] == it.kind());

                    let ty = if on_arrow {
                        // match foo { ..., pat => $0 }
                        cov_mark::hit!(expected_type_match_arm_body_without_leading_char);
                        cov_mark::hit!(expected_type_match_arm_body_with_leading_char);
                        sema.type_of_expr(&it.into())
                    } else {
                        // match foo { $0 }
                        cov_mark::hit!(expected_type_match_arm_without_leading_char);
                        it.expr().and_then(|e| sema.type_of_expr(&e))
                    }.map(TypeInfo::original);
                    (ty, None)
                },
                ast::IfExpr(it) => {
                    let ty = it.condition()
                        .and_then(|e| sema.type_of_expr(&e))
                        .map(TypeInfo::original);
                    (ty, None)
                },
                ast::IdentPat(it) => {
                    cov_mark::hit!(expected_type_if_let_with_leading_char);
                    cov_mark::hit!(expected_type_match_arm_with_leading_char);
                    let ty = sema.type_of_pat(&ast::Pat::from(it)).map(TypeInfo::original);
                    (ty, None)
                },
                ast::Fn(it) => {
                    cov_mark::hit!(expected_type_fn_ret_with_leading_char);
                    cov_mark::hit!(expected_type_fn_ret_without_leading_char);
                    let def = sema.to_def(&it);
                    (def.map(|def| def.ret_type(sema.db)), None)
                },
                ast::ClosureExpr(it) => {
                    let ty = sema.type_of_expr(&it.into());
                    ty.and_then(|ty| ty.original.as_callable(sema.db))
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
    };
    (ty.map(strip_refs), name)
}

fn classify_lifetime(
    _sema: &Semantics<'_, RootDatabase>,
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
    let lifetime = find_node_at_offset(original_file, lifetime.syntax().text_range().start());

    Some(LifetimeContext { lifetime, kind })
}

fn classify_name(
    sema: &Semantics<'_, RootDatabase>,
    original_file: &SyntaxNode,
    name: ast::Name,
) -> Option<NameContext> {
    let parent = name.syntax().parent()?;
    let kind = match_ast! {
        match parent {
            ast::Const(_) => NameKind::Const,
            ast::ConstParam(_) => NameKind::ConstParam,
            ast::Enum(_) => NameKind::Enum,
            ast::Fn(_) => NameKind::Function,
            ast::IdentPat(bind_pat) => {
                let mut pat_ctx = pattern_context_for(sema, original_file, bind_pat.into());
                if let Some(record_field) = ast::RecordPatField::for_field_name(&name) {
                    pat_ctx.record_pat = find_node_in_file_compensated(sema, original_file, &record_field.parent_record_pat());
                }

                NameKind::IdentPat(pat_ctx)
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
    let name = find_node_at_offset(original_file, name.syntax().text_range().start());
    Some(NameContext { name, kind })
}

fn classify_name_ref(
    sema: &Semantics<'_, RootDatabase>,
    original_file: &SyntaxNode,
    name_ref: ast::NameRef,
    parent: SyntaxNode,
) -> Option<(NameRefContext, QualifierCtx)> {
    let nameref = find_node_at_offset(original_file, name_ref.syntax().text_range().start());

    let make_res = |kind| (NameRefContext { nameref: nameref.clone(), kind }, Default::default());

    if let Some(record_field) = ast::RecordExprField::for_field_name(&name_ref) {
        let dot_prefix = previous_non_trivia_token(name_ref.syntax().clone())
            .map_or(false, |it| T![.] == it.kind());

        return find_node_in_file_compensated(
            sema,
            original_file,
            &record_field.parent_record_lit(),
        )
        .map(|expr| NameRefKind::RecordExpr { expr, dot_prefix })
        .map(make_res);
    }
    if let Some(record_field) = ast::RecordPatField::for_field_name_ref(&name_ref) {
        let kind = NameRefKind::Pattern(PatternContext {
            param_ctx: None,
            has_type_ascription: false,
            ref_token: None,
            mut_token: None,
            record_pat: find_node_in_file_compensated(
                sema,
                original_file,
                &record_field.parent_record_pat(),
            ),
            ..pattern_context_for(sema, original_file, record_field.parent_record_pat().into())
        });
        return Some(make_res(kind));
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

                let receiver_is_part_of_indivisible_expression = match &receiver {
                    Some(ast::Expr::IfExpr(_)) => {
                        let next_token_kind = next_non_trivia_token(name_ref.syntax().clone()).map(|t| t.kind());
                        next_token_kind == Some(SyntaxKind::ELSE_KW)
                    },
                    _ => false
                };
                if receiver_is_part_of_indivisible_expression {
                    return None;
                }

                let kind = NameRefKind::DotAccess(DotAccess {
                    receiver_ty: receiver.as_ref().and_then(|it| sema.type_of_expr(it)),
                    kind: DotAccessKind::Field { receiver_is_ambiguous_float_literal },
                    receiver
                });
                return Some(make_res(kind));
            },
            ast::MethodCallExpr(method) => {
                let receiver = find_opt_node_in_file(original_file, method.receiver());
                let kind = NameRefKind::DotAccess(DotAccess {
                    receiver_ty: receiver.as_ref().and_then(|it| sema.type_of_expr(it)),
                    kind: DotAccessKind::Method { has_parens: method.arg_list().map_or(false, |it| it.l_paren_token().is_some()) },
                    receiver
                });
                return Some(make_res(kind));
            },
            _ => return None,
        }
    };

    let path = segment.parent_path();
    let original_path = find_node_in_file_compensated(sema, original_file, &path);

    let mut path_ctx = PathCompletionCtx {
        has_call_parens: false,
        has_macro_bang: false,
        qualified: Qualified::No,
        parent: None,
        path: path.clone(),
        original_path,
        kind: PathKind::Item { kind: ItemListKind::SourceFile },
        has_type_args: false,
        use_tree_parent: false,
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
            find_node_in_file_compensated(sema, original_file, &record_expr)
        } else {
            None
        }
    };
    let after_if_expr = |node: SyntaxNode| {
        let prev_expr = (|| {
            let node = match node.parent().and_then(ast::ExprStmt::cast) {
                Some(stmt) => stmt.syntax().clone(),
                None => node,
            };
            let prev_sibling = non_trivia_sibling(node.into(), Direction::Prev)?.into_node()?;

            ast::ExprStmt::cast(prev_sibling.clone())
                .and_then(|it| it.expr())
                .or_else(|| ast::Expr::cast(prev_sibling))
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
                    ast::Item::Const(it) => it.body().is_none() && it.semicolon_token().is_none(),
                    ast::Item::Enum(it) => it.variant_list().is_none(),
                    ast::Item::ExternBlock(it) => it.extern_item_list().is_none(),
                    ast::Item::Fn(it) => it.body().is_none() && it.semicolon_token().is_none(),
                    ast::Item::Impl(it) => it.assoc_item_list().is_none(),
                    ast::Item::Module(it) => {
                        it.item_list().is_none() && it.semicolon_token().is_none()
                    }
                    ast::Item::Static(it) => it.body().is_none(),
                    ast::Item::Struct(it) => {
                        it.field_list().is_none() && it.semicolon_token().is_none()
                    }
                    ast::Item::Trait(it) => it.assoc_item_list().is_none(),
                    ast::Item::TypeAlias(it) => it.ty().is_none() && it.semicolon_token().is_none(),
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

    let type_location = |node: &SyntaxNode| {
        let parent = node.parent()?;
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
                ast::Impl(it) => {
                    match it.trait_() {
                        Some(t) if t.syntax() == node => TypeLocation::ImplTrait,
                        _ => match it.self_ty() {
                            Some(t) if t.syntax() == node => TypeLocation::ImplTarget,
                            _ => return None,
                        },
                    }
                },
                ast::TypeBound(_) => TypeLocation::TypeBound,
                // is this case needed?
                ast::TypeBoundList(_) => TypeLocation::TypeBound,
                ast::GenericArg(it) => TypeLocation::GenericArgList(find_opt_node_in_file_compensated(sema, original_file, it.syntax().parent().and_then(ast::GenericArgList::cast))),
                // is this case needed?
                ast::GenericArgList(it) => TypeLocation::GenericArgList(find_opt_node_in_file_compensated(sema, original_file, Some(it))),
                ast::TupleField(_) => TypeLocation::TupleField,
                _ => return None,
            }
        };
        Some(res)
    };

    let is_in_condition = |it: &ast::Expr| {
        (|| {
            let parent = it.syntax().parent()?;
            if let Some(expr) = ast::WhileExpr::cast(parent.clone()) {
                Some(expr.condition()? == *it)
            } else if let Some(expr) = ast::IfExpr::cast(parent) {
                Some(expr.condition()? == *it)
            } else {
                None
            }
        })()
        .unwrap_or(false)
    };

    let make_path_kind_expr = |expr: ast::Expr| {
        let it = expr.syntax();
        let in_block_expr = is_in_block(it);
        let in_loop_body = is_in_loop_body(it);
        let after_if_expr = after_if_expr(it.clone());
        let ref_expr_parent =
            path.as_single_name_ref().and_then(|_| it.parent()).and_then(ast::RefExpr::cast);
        let (innermost_ret_ty, self_param) = {
            let find_ret_ty = |it: SyntaxNode| {
                if let Some(item) = ast::Item::cast(it.clone()) {
                    match item {
                        ast::Item::Fn(f) => Some(sema.to_def(&f).map(|it| it.ret_type(sema.db))),
                        ast::Item::MacroCall(_) => None,
                        _ => Some(None),
                    }
                } else {
                    let expr = ast::Expr::cast(it)?;
                    let callable = match expr {
                        // FIXME
                        // ast::Expr::BlockExpr(b) if b.async_token().is_some() || b.try_token().is_some() => sema.type_of_expr(b),
                        ast::Expr::ClosureExpr(_) => sema.type_of_expr(&expr),
                        _ => return None,
                    };
                    Some(
                        callable
                            .and_then(|c| c.adjusted().as_callable(sema.db))
                            .map(|it| it.return_type()),
                    )
                }
            };
            let find_fn_self_param = |it| match it {
                ast::Item::Fn(fn_) => Some(sema.to_def(&fn_).and_then(|it| it.self_param(sema.db))),
                ast::Item::MacroCall(_) => None,
                _ => Some(None),
            };

            match find_node_in_file_compensated(sema, original_file, &expr) {
                Some(it) => {
                    let innermost_ret_ty = sema
                        .ancestors_with_macros(it.syntax().clone())
                        .find_map(find_ret_ty)
                        .flatten();

                    let self_param = sema
                        .ancestors_with_macros(it.syntax().clone())
                        .filter_map(ast::Item::cast)
                        .find_map(find_fn_self_param)
                        .flatten();
                    (innermost_ret_ty, self_param)
                }
                None => (None, None),
            }
        };
        let is_func_update = func_update_record(it);
        let in_condition = is_in_condition(&expr);
        let incomplete_let = it
            .parent()
            .and_then(ast::LetStmt::cast)
            .map_or(false, |it| it.semicolon_token().is_none());
        let impl_ = fetch_immediate_impl(sema, original_file, expr.syntax());

        let in_match_guard = match it.parent().and_then(ast::MatchArm::cast) {
            Some(arm) => arm
                .fat_arrow_token()
                .map_or(true, |arrow| it.text_range().start() < arrow.text_range().start()),
            None => false,
        };

        PathKind::Expr {
            expr_ctx: ExprCtx {
                in_block_expr,
                in_loop_body,
                after_if_expr,
                in_condition,
                ref_expr_parent,
                is_func_update,
                innermost_ret_ty,
                self_param,
                incomplete_let,
                impl_,
                in_match_guard,
            },
        }
    };
    let make_path_kind_type = |ty: ast::Type| {
        let location = type_location(ty.syntax());
        PathKind::Type { location: location.unwrap_or(TypeLocation::Other) }
    };

    let mut kind_macro_call = |it: ast::MacroCall| {
        path_ctx.has_macro_bang = it.excl_token().is_some();
        let parent = it.syntax().parent()?;
        // Any path in an item list will be treated as a macro call by the parser
        let kind = match_ast! {
            match parent {
                ast::MacroExpr(expr) => make_path_kind_expr(expr.into()),
                ast::MacroPat(it) => PathKind::Pat { pat_ctx: pattern_context_for(sema, original_file, it.into())},
                ast::MacroType(ty) => make_path_kind_type(ty.into()),
                ast::ItemList(_) => PathKind::Item { kind: ItemListKind::Module },
                ast::AssocItemList(_) => PathKind::Item { kind: match parent.parent() {
                    Some(it) => match_ast! {
                        match it {
                            ast::Trait(_) => ItemListKind::Trait,
                            ast::Impl(it) => if it.trait_().is_some() {
                                ItemListKind::TraitImpl(find_node_in_file_compensated(sema, original_file, &it))
                            } else {
                                ItemListKind::Impl
                            },
                            _ => return None
                        }
                    },
                    None => return None,
                } },
                ast::ExternItemList(_) => PathKind::Item { kind: ItemListKind::ExternBlock },
                ast::SourceFile(_) => PathKind::Item { kind: ItemListKind::SourceFile },
                _ => return None,
            }
        };
        Some(kind)
    };
    let make_path_kind_attr = |meta: ast::Meta| {
        let attr = meta.parent_attr()?;
        let kind = attr.kind();
        let attached = attr.syntax().parent()?;
        let is_trailing_outer_attr = kind != AttrKind::Inner
            && non_trivia_sibling(attr.syntax().clone().into(), syntax::Direction::Next).is_none();
        let annotated_item_kind = if is_trailing_outer_attr { None } else { Some(attached.kind()) };
        Some(PathKind::Attr { attr_ctx: AttrCtx { kind, annotated_item_kind } })
    };

    // Infer the path kind
    let parent = path.syntax().parent()?;
    let kind = match_ast! {
        match parent {
            ast::PathType(it) => make_path_kind_type(it.into()),
            ast::PathExpr(it) => {
                if let Some(p) = it.syntax().parent() {
                    if ast::ExprStmt::can_cast(p.kind()) {
                        if let Some(kind) = inbetween_body_and_decl_check(p) {
                            return Some(make_res(NameRefKind::Keyword(kind)));
                        }
                    }
                }

                path_ctx.has_call_parens = it.syntax().parent().map_or(false, |it| ast::CallExpr::can_cast(it.kind()));

                make_path_kind_expr(it.into())
            },
            ast::TupleStructPat(it) => {
                path_ctx.has_call_parens = true;
                PathKind::Pat { pat_ctx: pattern_context_for(sema, original_file, it.into()) }
            },
            ast::RecordPat(it) => {
                path_ctx.has_call_parens = true;
                PathKind::Pat { pat_ctx: pattern_context_for(sema, original_file, it.into()) }
            },
            ast::PathPat(it) => {
                PathKind::Pat { pat_ctx: pattern_context_for(sema, original_file, it.into())}
            },
            ast::MacroCall(it) => {
                // A macro call in this position is usually a result of parsing recovery, so check that
                if let Some(kind) = inbetween_body_and_decl_check(it.syntax().clone()) {
                    return Some(make_res(NameRefKind::Keyword(kind)));
                }

                kind_macro_call(it)?
            },
            ast::Meta(meta) => make_path_kind_attr(meta)?,
            ast::Visibility(it) => PathKind::Vis { has_in_token: it.in_token().is_some() },
            ast::UseTree(_) => PathKind::Use,
            // completing inside a qualifier
            ast::Path(parent) => {
                path_ctx.parent = Some(parent.clone());
                let parent = iter::successors(Some(parent), |it| it.parent_path()).last()?.syntax().parent()?;
                match_ast! {
                    match parent {
                        ast::PathType(it) => make_path_kind_type(it.into()),
                        ast::PathExpr(it) => {
                            path_ctx.has_call_parens = it.syntax().parent().map_or(false, |it| ast::CallExpr::can_cast(it.kind()));

                            make_path_kind_expr(it.into())
                        },
                        ast::TupleStructPat(it) => {
                            path_ctx.has_call_parens = true;
                            PathKind::Pat { pat_ctx: pattern_context_for(sema, original_file, it.into()) }
                        },
                        ast::RecordPat(it) => {
                            path_ctx.has_call_parens = true;
                            PathKind::Pat { pat_ctx: pattern_context_for(sema, original_file, it.into()) }
                        },
                        ast::PathPat(it) => {
                            PathKind::Pat { pat_ctx: pattern_context_for(sema, original_file, it.into())}
                        },
                        ast::MacroCall(it) => {
                            kind_macro_call(it)?
                        },
                        ast::Meta(meta) => make_path_kind_attr(meta)?,
                        ast::Visibility(it) => PathKind::Vis { has_in_token: it.in_token().is_some() },
                        ast::UseTree(_) => PathKind::Use,
                        ast::RecordExpr(it) => make_path_kind_expr(it.into()),
                        _ => return None,
                    }
                }
            },
            ast::RecordExpr(it) => make_path_kind_expr(it.into()),
            _ => return None,
        }
    };

    path_ctx.kind = kind;
    path_ctx.has_type_args = segment.generic_arg_list().is_some();

    // calculate the qualifier context
    if let Some((qualifier, use_tree_parent)) = path_or_use_tree_qualifier(&path) {
        path_ctx.use_tree_parent = use_tree_parent;
        if !use_tree_parent && segment.coloncolon_token().is_some() {
            path_ctx.qualified = Qualified::Absolute;
        } else {
            let qualifier = qualifier
                .segment()
                .and_then(|it| find_node_in_file(original_file, &it))
                .map(|it| it.parent_path());
            if let Some(qualifier) = qualifier {
                let type_anchor = match qualifier.segment().and_then(|it| it.kind()) {
                    Some(ast::PathSegmentKind::Type { type_ref: Some(type_ref), trait_ref })
                        if qualifier.qualifier().is_none() =>
                    {
                        Some((type_ref, trait_ref))
                    }
                    _ => None,
                };

                path_ctx.qualified = if let Some((ty, trait_ref)) = type_anchor {
                    let ty = match ty {
                        ast::Type::InferType(_) => None,
                        ty => sema.resolve_type(&ty),
                    };
                    let trait_ = trait_ref.and_then(|it| sema.resolve_trait(&it.path()?));
                    Qualified::TypeAnchor { ty, trait_ }
                } else {
                    let res = sema.resolve_path(&qualifier);

                    // For understanding how and why super_chain_len is calculated the way it
                    // is check the documentation at it's definition
                    let mut segment_count = 0;
                    let super_count = iter::successors(Some(qualifier.clone()), |p| p.qualifier())
                        .take_while(|p| {
                            p.segment()
                                .and_then(|s| {
                                    segment_count += 1;
                                    s.super_token()
                                })
                                .is_some()
                        })
                        .count();

                    let super_chain_len =
                        if segment_count > super_count { None } else { Some(super_count) };

                    Qualified::With { path: qualifier, resolution: res, super_chain_len }
                }
            };
        }
    } else if let Some(segment) = path.segment() {
        if segment.coloncolon_token().is_some() {
            path_ctx.qualified = Qualified::Absolute;
        }
    }

    let mut qualifier_ctx = QualifierCtx::default();
    if path_ctx.is_trivial_path() {
        // fetch the full expression that may have qualifiers attached to it
        let top_node = match path_ctx.kind {
            PathKind::Expr { expr_ctx: ExprCtx { in_block_expr: true, .. } } => {
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
                    qualifier_ctx.vis_node = error_node.children().find_map(ast::Visibility::cast);
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
                                // list
                                return None;
                            }
                        }
                    }
                }
            }
        }
    }
    Some((NameRefContext { nameref, kind: NameRefKind::Path(path_ctx) }, qualifier_ctx))
}

fn pattern_context_for(
    sema: &Semantics<'_, RootDatabase>,
    original_file: &SyntaxNode,
    pat: ast::Pat,
) -> PatternContext {
    let mut param_ctx = None;

    let mut missing_variants = vec![];

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
                        param_ctx = (|| {
                            let fake_param_list = param.syntax().parent().and_then(ast::ParamList::cast)?;
                            let param_list = find_node_in_file_compensated(sema, original_file, &fake_param_list)?;
                            let param_list_owner = param_list.syntax().parent()?;
                            let kind = match_ast! {
                                match param_list_owner {
                                    ast::ClosureExpr(closure) => ParamKind::Closure(closure),
                                    ast::Fn(fn_) => ParamKind::Function(fn_),
                                    _ => return None,
                                }
                            };
                            Some(ParamContext {
                                param_list, param, kind
                            })
                        })();
                        return (PatternRefutability::Irrefutable, has_type_ascription)
                    },
                    ast::MatchArm(match_arm) => {
                       let missing_variants_opt = match_arm
                            .syntax()
                            .parent()
                            .and_then(ast::MatchArmList::cast)
                            .and_then(|match_arm_list| {
                                match_arm_list
                                .syntax()
                                .parent()
                                .and_then(ast::MatchExpr::cast)
                                .and_then(|match_expr| {
                                    let expr_opt = find_opt_node_in_file(&original_file, match_expr.expr());

                                    expr_opt.and_then(|expr| {
                                        sema.type_of_expr(&expr)?
                                        .adjusted()
                                        .autoderef(sema.db)
                                        .find_map(|ty| match ty.as_adt() {
                                            Some(hir::Adt::Enum(e)) => Some(e),
                                            _ => None,
                                        }).and_then(|enum_| {
                                            Some(enum_.variants(sema.db))
                                        })
                                    })
                                }).and_then(|variants| {
                                   Some(variants.iter().filter_map(|variant| {
                                        let variant_name = variant.name(sema.db).display(sema.db).to_string();

                                        let variant_already_present = match_arm_list.arms().any(|arm| {
                                            arm.pat().and_then(|pat| {
                                                let pat_already_present = pat.syntax().to_string().contains(&variant_name);
                                                pat_already_present.then(|| pat_already_present)
                                            }).is_some()
                                        });

                                        (!variant_already_present).then_some(variant.clone())
                                    }).collect::<Vec<Variant>>())
                                })
                        });

                        if let Some(missing_variants_) = missing_variants_opt {
                            missing_variants = missing_variants_;
                        };

                        PatternRefutability::Refutable
                    },
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
        param_ctx,
        has_type_ascription,
        parent_pat: pat.syntax().parent().and_then(ast::Pat::cast),
        mut_token,
        ref_token,
        record_pat: None,
        impl_: fetch_immediate_impl(sema, original_file, pat.syntax()),
        missing_variants,
    }
}

fn fetch_immediate_impl(
    sema: &Semantics<'_, RootDatabase>,
    original_file: &SyntaxNode,
    node: &SyntaxNode,
) -> Option<ast::Impl> {
    let mut ancestors = ancestors_in_file_compensated(sema, original_file, node)?
        .filter_map(ast::Item::cast)
        .filter(|it| !matches!(it, ast::Item::MacroCall(_)));

    match ancestors.next()? {
        ast::Item::Const(_) | ast::Item::Fn(_) | ast::Item::TypeAlias(_) => (),
        ast::Item::Impl(it) => return Some(it),
        _ => return None,
    }
    match ancestors.next()? {
        ast::Item::Impl(it) => Some(it),
        _ => None,
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
fn find_node_in_file_compensated<N: AstNode>(
    sema: &Semantics<'_, RootDatabase>,
    in_file: &SyntaxNode,
    node: &N,
) -> Option<N> {
    ancestors_in_file_compensated(sema, in_file, node.syntax())?.find_map(N::cast)
}

fn ancestors_in_file_compensated<'sema>(
    sema: &'sema Semantics<'_, RootDatabase>,
    in_file: &SyntaxNode,
    node: &SyntaxNode,
) -> Option<impl Iterator<Item = SyntaxNode> + 'sema> {
    let syntax_range = in_file.text_range();
    let range = node.text_range();
    let end = range.end().checked_sub(TextSize::try_from(COMPLETION_MARKER.len()).ok()?)?;
    if end < range.start() {
        return None;
    }
    let range = TextRange::new(range.start(), end);
    // our inserted ident could cause `range` to go outside of the original syntax, so cap it
    let intersection = range.intersect(syntax_range)?;
    let node = match in_file.covering_element(intersection) {
        NodeOrToken::Node(node) => node,
        NodeOrToken::Token(tok) => tok.parent()?,
    };
    Some(sema.ancestors_with_macros(node))
}

/// Attempts to find `node` inside `syntax` via `node`'s text range while compensating
/// for the offset introduced by the fake ident..
/// This is wrong if `node` comes before the insertion point! Use `find_node_in_file` instead.
fn find_opt_node_in_file_compensated<N: AstNode>(
    sema: &Semantics<'_, RootDatabase>,
    syntax: &SyntaxNode,
    node: Option<N>,
) -> Option<N> {
    find_node_in_file_compensated(sema, syntax, &node?)
}

fn path_or_use_tree_qualifier(path: &ast::Path) -> Option<(ast::Path, bool)> {
    if let Some(qual) = path.qualifier() {
        return Some((qual, false));
    }
    let use_tree_list = path.syntax().ancestors().find_map(ast::UseTreeList::cast)?;
    let use_tree = use_tree_list.syntax().parent().and_then(ast::UseTree::cast)?;
    Some((use_tree.path()?, true))
}

fn is_in_token_of_for_loop(path: &ast::Path) -> bool {
    // oh my ...
    (|| {
        let expr = path.syntax().parent().and_then(ast::PathExpr::cast)?;
        let for_expr = expr.syntax().parent().and_then(ast::ForExpr::cast)?;
        if for_expr.in_token().is_some() {
            return Some(false);
        }
        let pat = for_expr.pat()?;
        let next_sibl = next_non_trivia_sibling(pat.syntax().clone().into())?;
        Some(match next_sibl {
            syntax::NodeOrToken::Node(n) => {
                n.text_range().start() == path.syntax().text_range().start()
            }
            syntax::NodeOrToken::Token(t) => {
                t.text_range().start() == path.syntax().text_range().start()
            }
        })
    })()
    .unwrap_or(false)
}

fn is_in_loop_body(node: &SyntaxNode) -> bool {
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

fn previous_non_trivia_token(e: impl Into<SyntaxElement>) -> Option<SyntaxToken> {
    let mut token = match e.into() {
        SyntaxElement::Node(n) => n.first_token()?,
        SyntaxElement::Token(t) => t,
    }
    .prev_token();
    while let Some(inner) = token {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            token = inner.prev_token();
        }
    }
    None
}

fn next_non_trivia_token(e: impl Into<SyntaxElement>) -> Option<SyntaxToken> {
    let mut token = match e.into() {
        SyntaxElement::Node(n) => n.last_token()?,
        SyntaxElement::Token(t) => t,
    }
    .next_token();
    while let Some(inner) = token {
        if !inner.kind().is_trivia() {
            return Some(inner);
        } else {
            token = inner.next_token();
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
