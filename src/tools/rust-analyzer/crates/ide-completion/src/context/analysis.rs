//! Module responsible for analyzing the code surrounding the cursor for completion.
use std::iter;

use hir::{ExpandResult, InFile, Semantics, Type, TypeInfo, Variant};
use ide_db::{
    RootDatabase, active_parameter::ActiveParameter, syntax_helpers::node_ext::find_loops,
};
use itertools::{Either, Itertools};
use stdx::always;
use syntax::{
    AstNode, AstToken, Direction, NodeOrToken, SyntaxElement, SyntaxKind, SyntaxNode, SyntaxToken,
    T, TextRange, TextSize,
    algo::{
        self, ancestors_at_offset, find_node_at_offset, non_trivia_sibling,
        previous_non_trivia_token,
    },
    ast::{
        self, AttrKind, HasArgList, HasGenericArgs, HasGenericParams, HasLoopBody, HasName,
        NameOrNameRef,
    },
    match_ast,
};

use crate::{
    completions::postfix::is_in_condition,
    context::{
        AttrCtx, BreakableKind, COMPLETION_MARKER, CompletionAnalysis, DotAccess, DotAccessExprCtx,
        DotAccessKind, ItemListKind, LifetimeContext, LifetimeKind, NameContext, NameKind,
        NameRefContext, NameRefKind, ParamContext, ParamKind, PathCompletionCtx, PathExprCtx,
        PathKind, PatternContext, PatternRefutability, Qualified, QualifierCtx,
        TypeAscriptionTarget, TypeLocation,
    },
};

#[derive(Debug)]
struct ExpansionResult {
    original_file: SyntaxNode,
    speculative_file: SyntaxNode,
    /// The offset in the original file.
    original_offset: TextSize,
    /// The offset in the speculatively expanded file.
    speculative_offset: TextSize,
    fake_ident_token: SyntaxToken,
    derive_ctx: Option<(SyntaxNode, SyntaxNode, TextSize, ast::Attr)>,
}

pub(super) struct AnalysisResult<'db> {
    pub(super) analysis: CompletionAnalysis<'db>,
    pub(super) expected: (Option<Type<'db>>, Option<ast::NameOrNameRef>),
    pub(super) qualifier_ctx: QualifierCtx,
    /// the original token of the expanded file
    pub(super) token: SyntaxToken,
    /// The offset in the original file.
    pub(super) original_offset: TextSize,
}

pub(super) fn expand_and_analyze<'db>(
    sema: &Semantics<'db, RootDatabase>,
    original_file: InFile<SyntaxNode>,
    speculative_file: SyntaxNode,
    offset: TextSize,
    original_token: &SyntaxToken,
) -> Option<AnalysisResult<'db>> {
    // as we insert after the offset, right biased will *always* pick the identifier no matter
    // if there is an ident already typed or not
    let fake_ident_token = speculative_file.token_at_offset(offset).right_biased()?;
    // the relative offset between the cursor and the *identifier* token we are completing on
    let relative_offset = offset - fake_ident_token.text_range().start();
    // make the offset point to the start of the original token, as that is what the
    // intermediate offsets calculated in expansion always points to
    let offset = offset - relative_offset;
    let expansion = expand_maybe_stop(
        sema,
        original_file.clone(),
        speculative_file.clone(),
        offset,
        fake_ident_token.clone(),
        relative_offset,
    )
    .unwrap_or(ExpansionResult {
        original_file: original_file.value,
        speculative_file,
        original_offset: offset,
        speculative_offset: fake_ident_token.text_range().start(),
        fake_ident_token,
        derive_ctx: None,
    });

    // add the relative offset back, so that left_biased finds the proper token
    let original_offset = expansion.original_offset + relative_offset;
    let token = expansion.original_file.token_at_offset(original_offset).left_biased()?;

    analyze(sema, expansion, original_token, &token).map(|(analysis, expected, qualifier_ctx)| {
        AnalysisResult { analysis, expected, qualifier_ctx, token, original_offset }
    })
}

fn token_at_offset_ignore_whitespace(file: &SyntaxNode, offset: TextSize) -> Option<SyntaxToken> {
    let token = file.token_at_offset(offset).left_biased()?;
    algo::skip_whitespace_token(token, Direction::Prev)
}

/// Expand attributes and macro calls at the current cursor position for both the original file
/// and fake file repeatedly. As soon as one of the two expansions fail we stop so the original
/// and speculative states stay in sync.
///
/// We do this by recursively expanding all macros and picking the best possible match. We cannot just
/// choose the first expansion each time because macros can expand to something that does not include
/// our completion marker, e.g.:
///
/// ```ignore
/// macro_rules! helper { ($v:ident) => {} }
/// macro_rules! my_macro {
///     ($v:ident) => {
///         helper!($v);
///         $v
///     };
/// }
///
/// my_macro!(complete_me_here);
/// ```
/// If we would expand the first thing we encounter only (which in fact this method used to do), we would
/// be unable to complete here, because we would be walking directly into the void. So we instead try
/// *every* possible path.
///
/// This can also creates discrepancies between the speculative and real expansions: because we insert
/// tokens, we insert characters, which means if we try the second occurrence it may not be at the same
/// position in the original and speculative file. We take an educated guess here, and for each token
/// that we check, we subtract `COMPLETION_MARKER.len()`. This may not be accurate because proc macros
/// can insert the text of the completion marker in other places while removing the span, but this is
/// the best we can do.
fn expand_maybe_stop(
    sema: &Semantics<'_, RootDatabase>,
    original_file: InFile<SyntaxNode>,
    speculative_file: SyntaxNode,
    original_offset: TextSize,
    fake_ident_token: SyntaxToken,
    relative_offset: TextSize,
) -> Option<ExpansionResult> {
    if let result @ Some(_) = expand(
        sema,
        original_file.clone(),
        speculative_file.clone(),
        original_offset,
        fake_ident_token.clone(),
        relative_offset,
    ) {
        return result;
    }

    // We can't check whether the fake expansion is inside macro call, because that requires semantic info.
    // But hopefully checking just the real one should be enough.
    if token_at_offset_ignore_whitespace(&original_file.value, original_offset + relative_offset)
        .is_some_and(|original_token| {
            !sema.is_inside_macro_call(original_file.with_value(&original_token))
        })
    {
        // Recursion base case.
        Some(ExpansionResult {
            original_file: original_file.value,
            speculative_file,
            original_offset,
            speculative_offset: fake_ident_token.text_range().start(),
            fake_ident_token,
            derive_ctx: None,
        })
    } else {
        None
    }
}

fn expand(
    sema: &Semantics<'_, RootDatabase>,
    original_file: InFile<SyntaxNode>,
    speculative_file: SyntaxNode,
    original_offset: TextSize,
    fake_ident_token: SyntaxToken,
    relative_offset: TextSize,
) -> Option<ExpansionResult> {
    let _p = tracing::info_span!("CompletionContext::expand").entered();

    let parent_item =
        |item: &ast::Item| item.syntax().ancestors().skip(1).find_map(ast::Item::cast);
    let original_node = token_at_offset_ignore_whitespace(&original_file.value, original_offset)
        .and_then(|token| token.parent_ancestors().find_map(ast::Item::cast));
    let ancestor_items = iter::successors(
        Option::zip(
            original_node,
            find_node_at_offset::<ast::Item>(
                &speculative_file,
                fake_ident_token.text_range().start(),
            ),
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
            (
                Some(ExpandResult { value: actual_expansion, err: _ }),
                Some((fake_expansion, fake_mapped_tokens)),
            ) => {
                let mut accumulated_offset_from_fake_tokens = 0;
                let actual_range = actual_expansion.text_range().end();
                let result = fake_mapped_tokens
                    .into_iter()
                    .filter_map(|(fake_mapped_token, rank)| {
                        let accumulated_offset = accumulated_offset_from_fake_tokens;
                        if !fake_mapped_token.text().contains(COMPLETION_MARKER) {
                            // Proc macros can make the same span with different text, we don't
                            // want them to participate in completion because the macro author probably
                            // didn't intend them to.
                            return None;
                        }
                        accumulated_offset_from_fake_tokens += COMPLETION_MARKER.len();

                        let new_offset = fake_mapped_token.text_range().start()
                            - TextSize::new(accumulated_offset as u32);
                        if new_offset + relative_offset > actual_range {
                            // offset outside of bounds from the original expansion,
                            // stop here to prevent problems from happening
                            return None;
                        }
                        let result = expand_maybe_stop(
                            sema,
                            actual_expansion.clone(),
                            fake_expansion.clone(),
                            new_offset,
                            fake_mapped_token,
                            relative_offset,
                        )?;
                        Some((result, rank))
                    })
                    .min_by_key(|(_, rank)| *rank)
                    .map(|(result, _)| result);
                if result.is_some() {
                    return result;
                }
            }
            // exactly one expansion failed, inconsistent state so stop expanding completely
            _ => break 'ancestors,
        }
    }

    // No attributes have been expanded, so look for macro_call! token trees or derive token trees
    let orig_tt = ancestors_at_offset(&original_file.value, original_offset)
        .map_while(Either::<ast::TokenTree, ast::Meta>::cast)
        .last()?;
    let spec_tt = ancestors_at_offset(&speculative_file, fake_ident_token.text_range().start())
        .map_while(Either::<ast::TokenTree, ast::Meta>::cast)
        .last()?;

    let (tts, attrs) = match (orig_tt, spec_tt) {
        (Either::Left(orig_tt), Either::Left(spec_tt)) => {
            let attrs = orig_tt
                .syntax()
                .parent()
                .and_then(ast::Meta::cast)
                .and_then(|it| it.parent_attr())
                .zip(
                    spec_tt
                        .syntax()
                        .parent()
                        .and_then(ast::Meta::cast)
                        .and_then(|it| it.parent_attr()),
                );
            (Some((orig_tt, spec_tt)), attrs)
        }
        (Either::Right(orig_path), Either::Right(spec_path)) => {
            (None, orig_path.parent_attr().zip(spec_path.parent_attr()))
        }
        _ => return None,
    };

    // Expand pseudo-derive expansion aka `derive(Debug$0)`
    if let Some((orig_attr, spec_attr)) = attrs {
        if let (Some(actual_expansion), Some((fake_expansion, fake_mapped_tokens))) = (
            sema.expand_derive_as_pseudo_attr_macro(&orig_attr),
            sema.speculative_expand_derive_as_pseudo_attr_macro(
                &orig_attr,
                &spec_attr,
                fake_ident_token.clone(),
            ),
        ) && let Some((fake_mapped_token, _)) =
            fake_mapped_tokens.into_iter().min_by_key(|(_, rank)| *rank)
        {
            return Some(ExpansionResult {
                original_file: original_file.value,
                speculative_file,
                original_offset,
                speculative_offset: fake_ident_token.text_range().start(),
                fake_ident_token,
                derive_ctx: Some((
                    actual_expansion,
                    fake_expansion,
                    fake_mapped_token.text_range().start(),
                    orig_attr,
                )),
            });
        }

        if let Some(spec_adt) =
            spec_attr.syntax().ancestors().find_map(ast::Item::cast).and_then(|it| match it {
                ast::Item::Struct(it) => Some(ast::Adt::Struct(it)),
                ast::Item::Enum(it) => Some(ast::Adt::Enum(it)),
                ast::Item::Union(it) => Some(ast::Adt::Union(it)),
                _ => None,
            })
        {
            // might be the path of derive helper or a token tree inside of one
            if let Some(helpers) = sema.derive_helper(&orig_attr) {
                for (_mac, file) in helpers {
                    if let Some((fake_expansion, fake_mapped_tokens)) = sema.speculative_expand_raw(
                        file,
                        spec_adt.syntax(),
                        fake_ident_token.clone(),
                    ) {
                        // we are inside a derive helper token tree, treat this as being inside
                        // the derive expansion
                        let actual_expansion = sema.parse_or_expand(file.into());
                        let mut accumulated_offset_from_fake_tokens = 0;
                        let actual_range = actual_expansion.text_range().end();
                        let result = fake_mapped_tokens
                            .into_iter()
                            .filter_map(|(fake_mapped_token, rank)| {
                                let accumulated_offset = accumulated_offset_from_fake_tokens;
                                if !fake_mapped_token.text().contains(COMPLETION_MARKER) {
                                    // Proc macros can make the same span with different text, we don't
                                    // want them to participate in completion because the macro author probably
                                    // didn't intend them to.
                                    return None;
                                }
                                accumulated_offset_from_fake_tokens += COMPLETION_MARKER.len();

                                let new_offset = fake_mapped_token.text_range().start()
                                    - TextSize::new(accumulated_offset as u32);
                                if new_offset + relative_offset > actual_range {
                                    // offset outside of bounds from the original expansion,
                                    // stop here to prevent problems from happening
                                    return None;
                                }
                                let result = expand_maybe_stop(
                                    sema,
                                    InFile::new(file.into(), actual_expansion.clone()),
                                    fake_expansion.clone(),
                                    new_offset,
                                    fake_mapped_token,
                                    relative_offset,
                                )?;
                                Some((result, rank))
                            })
                            .min_by_key(|(_, rank)| *rank)
                            .map(|(result, _)| result);
                        if result.is_some() {
                            return result;
                        }
                    }
                }
            }
        }
        // at this point we won't have any more successful expansions, so stop
        return None;
    }

    // Expand fn-like macro calls
    let (orig_tt, spec_tt) = tts?;
    let (actual_macro_call, macro_call_with_fake_ident) = (
        orig_tt.syntax().parent().and_then(ast::MacroCall::cast)?,
        spec_tt.syntax().parent().and_then(ast::MacroCall::cast)?,
    );
    let mac_call_path0 = actual_macro_call.path().as_ref().map(|s| s.syntax().text());
    let mac_call_path1 = macro_call_with_fake_ident.path().as_ref().map(|s| s.syntax().text());

    // inconsistent state, stop expanding
    if mac_call_path0 != mac_call_path1 {
        return None;
    }
    let speculative_args = macro_call_with_fake_ident.token_tree()?;

    match (
        sema.expand_macro_call(&actual_macro_call),
        sema.speculative_expand_macro_call(&actual_macro_call, &speculative_args, fake_ident_token),
    ) {
        // successful expansions
        (Some(actual_expansion), Some((fake_expansion, fake_mapped_tokens))) => {
            let mut accumulated_offset_from_fake_tokens = 0;
            let actual_range = actual_expansion.text_range().end();
            fake_mapped_tokens
                .into_iter()
                .filter_map(|(fake_mapped_token, rank)| {
                    let accumulated_offset = accumulated_offset_from_fake_tokens;
                    if !fake_mapped_token.text().contains(COMPLETION_MARKER) {
                        // Proc macros can make the same span with different text, we don't
                        // want them to participate in completion because the macro author probably
                        // didn't intend them to.
                        return None;
                    }
                    accumulated_offset_from_fake_tokens += COMPLETION_MARKER.len();

                    let new_offset = fake_mapped_token.text_range().start()
                        - TextSize::new(accumulated_offset as u32);
                    if new_offset + relative_offset > actual_range {
                        // offset outside of bounds from the original expansion,
                        // stop here to prevent problems from happening
                        return None;
                    }
                    let result = expand_maybe_stop(
                        sema,
                        actual_expansion.clone(),
                        fake_expansion.clone(),
                        new_offset,
                        fake_mapped_token,
                        relative_offset,
                    )?;
                    Some((result, rank))
                })
                .min_by_key(|(_, rank)| *rank)
                .map(|(result, _)| result)
        }
        // at least one expansion failed, we won't have anything to expand from this point
        // onwards so break out
        _ => None,
    }
}

/// Fill the completion context, this is what does semantic reasoning about the surrounding context
/// of the completion location.
fn analyze<'db>(
    sema: &Semantics<'db, RootDatabase>,
    expansion_result: ExpansionResult,
    original_token: &SyntaxToken,
    self_token: &SyntaxToken,
) -> Option<(CompletionAnalysis<'db>, (Option<Type<'db>>, Option<ast::NameOrNameRef>), QualifierCtx)>
{
    let _p = tracing::info_span!("CompletionContext::analyze").entered();
    let ExpansionResult {
        original_file,
        speculative_file,
        original_offset: _,
        speculative_offset,
        fake_ident_token,
        derive_ctx,
    } = expansion_result;

    if original_token.kind() != self_token.kind()
        // FIXME: This check can be removed once we use speculative database forking for completions
        && !(original_token.kind().is_punct() || original_token.kind().is_trivia())
        && !(SyntaxKind::is_any_identifier(original_token.kind())
            && SyntaxKind::is_any_identifier(self_token.kind()))
    {
        return None;
    }

    // Overwrite the path kind for derives
    if let Some((original_file, file_with_fake_ident, offset, origin_attr)) = derive_ctx {
        if let Some(ast::NameLike::NameRef(name_ref)) =
            find_node_at_offset(&file_with_fake_ident, offset)
        {
            let parent = name_ref.syntax().parent()?;
            let (mut nameref_ctx, _) =
                classify_name_ref(sema, &original_file, name_ref, offset, parent)?;
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

    let Some(name_like) = find_node_at_offset(&speculative_file, speculative_offset) else {
        let analysis = if let Some(original) = ast::String::cast(original_token.clone()) {
            CompletionAnalysis::String { original, expanded: ast::String::cast(self_token.clone()) }
        } else {
            // Fix up trailing whitespace problem
            // #[attr(foo = $0
            let token = syntax::algo::skip_trivia_token(self_token.clone(), Direction::Prev)?;
            let p = token.parent()?;
            if p.kind() == SyntaxKind::TOKEN_TREE
                && p.ancestors().any(|it| it.kind() == SyntaxKind::META)
            {
                let colon_prefix = previous_non_trivia_token(self_token.clone())
                    .is_some_and(|it| T![:] == it.kind());

                CompletionAnalysis::UnexpandedAttrTT {
                    fake_attribute_under_caret: fake_ident_token
                        .parent_ancestors()
                        .find_map(ast::Attr::cast),
                    colon_prefix,
                    extern_crate: p.ancestors().find_map(ast::ExternCrate::cast),
                }
            } else if p.kind() == SyntaxKind::TOKEN_TREE
                && p.ancestors().any(|it| ast::Macro::can_cast(it.kind()))
            {
                if let Some([_ident, colon, _name, dollar]) = fake_ident_token
                    .siblings_with_tokens(Direction::Prev)
                    .filter(|it| !it.kind().is_trivia())
                    .take(4)
                    .collect_array()
                    && dollar.kind() == T![$]
                    && colon.kind() == T![:]
                {
                    CompletionAnalysis::MacroSegment
                } else {
                    return None;
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
            let (nameref_ctx, qualifier_ctx) = classify_name_ref(
                sema,
                &original_file,
                name_ref,
                expansion_result.original_offset,
                parent,
            )?;

            if let NameRefContext {
                kind:
                    NameRefKind::Path(PathCompletionCtx { kind: PathKind::Expr { .. }, path, .. }, ..),
                ..
            } = &nameref_ctx
                && is_in_token_of_for_loop(path)
            {
                // for pat $0
                // there is nothing to complete here except `in` keyword
                // don't bother populating the context
                // Ideally this special casing wouldn't be needed, but the parser recovers
                return None;
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
fn expected_type_and_name<'db>(
    sema: &Semantics<'db, RootDatabase>,
    self_token: &SyntaxToken,
    name_like: &ast::NameLike,
) -> (Option<Type<'db>>, Option<NameOrNameRef>) {
    let token = prev_special_biased_token_at_trivia(self_token.clone());
    let mut node = match token.parent() {
        Some(it) => it,
        None => return (None, None),
    };

    let strip_refs = |mut ty: Type<'db>| match name_like {
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
            let refs_level = top_syn
                .ancestors()
                .skip(1)
                .map_while(Either::<ast::RefExpr, ast::PrefixExpr>::cast)
                .take_while(|it| match it {
                    Either::Left(_) => true,
                    Either::Right(prefix) => prefix.op_kind() == Some(ast::UnaryOp::Deref),
                })
                .fold(0i32, |level, expr| match expr {
                    Either::Left(_) => level + 1,
                    Either::Right(_) => level - 1,
                });
            for _ in 0..refs_level {
                cov_mark::hit!(expected_type_fn_param_ref);
                ty = ty.strip_reference();
            }
            for _ in refs_level..0 {
                cov_mark::hit!(expected_type_fn_param_deref);
                ty = ty.add_reference(hir::Mutability::Shared);
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
                        .map(TypeInfo::original)
                        .filter(|ty| {
                            // don't infer the let type if the expr is a function,
                            // preventing parenthesis from vanishing
                            it.ty().is_some() || !ty.is_fn()
                        });
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
                ast::BinExpr(it) => {
                    if let Some(ast::BinaryOp::Assignment { op: None }) = it.op_kind() {
                        let ty = it.lhs()
                            .and_then(|lhs| sema.type_of_expr(&lhs))
                            .or_else(|| it.rhs().and_then(|rhs| sema.type_of_expr(&rhs)))
                            .map(TypeInfo::original);
                        (ty, None)
                    } else if let Some(ast::BinaryOp::LogicOp(_)) = it.op_kind() {
                        let ty = sema.type_of_expr(&it.clone().into()).map(TypeInfo::original);
                        (ty, None)
                    } else {
                        (None, None)
                    }
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
                            cov_mark::hit!(expected_type_struct_field_followed_by_comma);
                            let expr_field = previous_non_trivia_token(token.clone())?.parent().and_then(ast::RecordExprField::cast)?;
                            let (_, _, ty) = sema.resolve_record_field(&expr_field)?;
                            Some((
                                Some(ty),
                                expr_field.field_name().map(NameOrNameRef::NameRef),
                            ))
                        }
                    })().unwrap_or((None, None))
                },
                ast::RecordExprField(it) => {
                    let field_ty = sema.resolve_record_field(&it).map(|(_, _, ty)| ty);
                    let field_name = it.field_name().map(NameOrNameRef::NameRef);
                    if let Some(expr) = it.expr() {
                        cov_mark::hit!(expected_type_struct_field_with_leading_char);
                        let ty = field_ty
                            .or_else(|| sema.type_of_expr(&expr).map(TypeInfo::original));
                        (ty, field_name)
                    } else {
                        (field_ty, field_name)
                    }
                },
                // match foo { $0 }
                // match foo { ..., pat => $0 }
                ast::MatchExpr(it) => {
                    let on_arrow = previous_non_trivia_token(token.clone()).is_some_and(|it| T![=>] == it.kind());

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
                ast::MatchArm(it) => {
                    let on_arrow = previous_non_trivia_token(token.clone()).is_some_and(|it| T![=>] == it.kind());
                    let in_body = it.expr().is_some_and(|it| it.syntax().text_range().contains_range(token.text_range()));
                    let match_expr = it.parent_match();

                    let ty = if on_arrow || in_body {
                        // match foo { ..., pat => $0 }
                        cov_mark::hit!(expected_type_match_arm_body_without_leading_char);
                        cov_mark::hit!(expected_type_match_arm_body_with_leading_char);
                        sema.type_of_expr(&match_expr.into())
                    } else {
                        // match foo { $0 }
                        cov_mark::hit!(expected_type_match_arm_without_leading_char);
                        match_expr.expr().and_then(|e| sema.type_of_expr(&e))
                    }.map(TypeInfo::original);
                    (ty, None)
                },
                ast::IfExpr(it) => {
                    let ty = if let Some(body) = it.then_branch()
                        && token.text_range().end() > body.syntax().text_range().start()
                    {
                        sema.type_of_expr(&body.into())
                    } else {
                        it.condition().and_then(|e| sema.type_of_expr(&e))
                    }.map(TypeInfo::original);
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
                ast::ReturnExpr(it) => {
                    let fn_ = sema.ancestors_with_macros(it.syntax().clone())
                        .find_map(Either::<ast::Fn, ast::ClosureExpr>::cast);
                    let ty = fn_.and_then(|f| match f {
                        Either::Left(f) => Some(sema.to_def(&f)?.ret_type(sema.db)),
                        Either::Right(f) => {
                            let ty = sema.type_of_expr(&f.into())?.original.as_callable(sema.db)?;
                            Some(ty.return_type())
                        },
                    });
                    (ty, None)
                },
                ast::BreakExpr(it) => {
                    let ty = it.break_token()
                        .and_then(|it| find_loops(sema, &it)?.next())
                        .and_then(|expr| sema.type_of_expr(&expr));
                    (ty.map(TypeInfo::original), None)
                },
                ast::ClosureExpr(it) => {
                    let ty = sema.type_of_expr(&it.into());
                    ty.and_then(|ty| ty.original.as_callable(sema.db))
                        .map(|c| (Some(c.return_type()), None))
                        .unwrap_or((None, None))
                },
                ast::ParamList(it) => {
                    let closure = it.syntax().parent().and_then(ast::ClosureExpr::cast);
                    let ty = closure
                        .filter(|_| it.syntax().text_range().end() <= self_token.text_range().start())
                        .and_then(|it| sema.type_of_expr(&it.into()));
                    ty.and_then(|ty| ty.original.as_callable(sema.db))
                        .map(|c| (Some(c.return_type()), None))
                        .unwrap_or((None, None))
                },
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
    sema: &Semantics<'_, RootDatabase>,
    original_file: &SyntaxNode,
    lifetime: ast::Lifetime,
) -> Option<LifetimeContext> {
    let parent = lifetime.syntax().parent()?;
    if parent.kind() == SyntaxKind::ERROR {
        return None;
    }

    let lifetime =
        find_node_at_offset::<ast::Lifetime>(original_file, lifetime.syntax().text_range().start());
    let kind = match_ast! {
        match parent {
            ast::LifetimeParam(_) => LifetimeKind::LifetimeParam,
            ast::BreakExpr(_) => LifetimeKind::LabelRef,
            ast::ContinueExpr(_) => LifetimeKind::LabelRef,
            ast::Label(_) => LifetimeKind::LabelDef,
            _ => {
                let def = lifetime.as_ref().and_then(|lt| sema.scope(lt.syntax())?.generic_def());
                LifetimeKind::Lifetime { in_lifetime_param_bound: ast::TypeBound::can_cast(parent.kind()), def }
            },
        }
    };

    Some(LifetimeContext { kind })
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

fn classify_name_ref<'db>(
    sema: &Semantics<'db, RootDatabase>,
    original_file: &SyntaxNode,
    name_ref: ast::NameRef,
    original_offset: TextSize,
    parent: SyntaxNode,
) -> Option<(NameRefContext<'db>, QualifierCtx)> {
    let nameref = find_node_at_offset(original_file, original_offset);

    let make_res = |kind| (NameRefContext { nameref: nameref.clone(), kind }, Default::default());

    if let Some(record_field) = ast::RecordExprField::for_field_name(&name_ref) {
        let dot_prefix = previous_non_trivia_token(name_ref.syntax().clone())
            .is_some_and(|it| T![.] == it.kind());

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

    let field_expr_handle = |receiver, node| {
        let receiver = find_opt_node_in_file(original_file, receiver);
        let receiver_is_ambiguous_float_literal = match &receiver {
            Some(ast::Expr::Literal(l)) => matches! {
                l.kind(),
                ast::LiteralKind::FloatNumber { .. } if l.syntax().last_token().is_some_and(|it| it.text().ends_with('.'))
            },
            _ => false,
        };

        let receiver_is_part_of_indivisible_expression = match &receiver {
            Some(ast::Expr::IfExpr(_)) => {
                let next_token_kind =
                    next_non_trivia_token(name_ref.syntax().clone()).map(|t| t.kind());
                next_token_kind == Some(SyntaxKind::ELSE_KW)
            }
            _ => false,
        };
        if receiver_is_part_of_indivisible_expression {
            return None;
        }

        let mut receiver_ty = receiver.as_ref().and_then(|it| sema.type_of_expr(it));
        if receiver_is_ambiguous_float_literal {
            // `123.|` is parsed as a float but should actually be an integer.
            always!(receiver_ty.as_ref().is_none_or(|receiver_ty| receiver_ty.original.is_float()));
            receiver_ty =
                Some(TypeInfo { original: hir::BuiltinType::i32().ty(sema.db), adjusted: None });
        }

        let kind = NameRefKind::DotAccess(DotAccess {
            receiver_ty,
            kind: DotAccessKind::Field { receiver_is_ambiguous_float_literal },
            receiver,
            ctx: DotAccessExprCtx {
                in_block_expr: is_in_block(node),
                in_breakable: is_in_breakable(node).unzip().0,
            },
        });
        Some(make_res(kind))
    };

    let segment = match_ast! {
        match parent {
            ast::PathSegment(segment) => segment,
            ast::FieldExpr(field) => {
                return field_expr_handle(field.expr(), field.syntax());
            },
            ast::ExternCrate(_) => {
                let kind = NameRefKind::ExternCrate;
                return Some(make_res(kind));
            },
            ast::MethodCallExpr(method) => {
                let receiver = find_opt_node_in_file(original_file, method.receiver());
                let has_parens = has_parens(&method);
                if !has_parens && let Some(res) = field_expr_handle(method.receiver(), method.syntax()) {
                    return Some(res)
                }
                let kind = NameRefKind::DotAccess(DotAccess {
                    receiver_ty: receiver.as_ref().and_then(|it| sema.type_of_expr(it)),
                    kind: DotAccessKind::Method,
                    receiver,
                    ctx: DotAccessExprCtx { in_block_expr: is_in_block(method.syntax()), in_breakable: is_in_breakable(method.syntax()).unzip().0 }
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

    let func_update_record = |syn: &SyntaxNode| {
        if let Some(record_expr) = syn.ancestors().nth(2).and_then(ast::RecordExpr::cast) {
            find_node_in_file_compensated(sema, original_file, &record_expr)
        } else {
            None
        }
    };
    let prev_expr = |node: SyntaxNode| {
        let node = match node.parent().and_then(ast::ExprStmt::cast) {
            Some(stmt) => stmt.syntax().clone(),
            None => node,
        };
        let prev_sibling = non_trivia_sibling(node.into(), Direction::Prev)?.into_node()?;

        match_ast! {
            match prev_sibling {
                ast::ExprStmt(stmt) => stmt.expr().filter(|_| stmt.semicolon_token().is_none()),
                ast::LetStmt(stmt) => stmt.initializer().filter(|_| stmt.semicolon_token().is_none()),
                ast::Expr(expr) => Some(expr),
                _ => None,
            }
        }
    };
    let after_incomplete_let = |node: SyntaxNode| {
        prev_expr(node).and_then(|it| it.syntax().parent()).and_then(ast::LetStmt::cast)
    };
    let before_else_kw = |node: &SyntaxNode| {
        node.parent()
            .and_then(ast::ExprStmt::cast)
            .filter(|stmt| stmt.semicolon_token().is_none())
            .and_then(|stmt| non_trivia_sibling(stmt.syntax().clone().into(), Direction::Next))
            .and_then(NodeOrToken::into_node)
            .filter(|next| next.kind() == SyntaxKind::ERROR)
            .and_then(|next| next.first_token())
            .is_some_and(|token| token.kind() == SyntaxKind::ELSE_KW)
    };
    let is_in_value = |it: &SyntaxNode| {
        let Some(node) = it.parent() else { return false };
        let kind = node.kind();
        ast::LetStmt::can_cast(kind)
            || ast::ArgList::can_cast(kind)
            || ast::ArrayExpr::can_cast(kind)
            || ast::ParenExpr::can_cast(kind)
            || ast::BreakExpr::can_cast(kind)
            || ast::ReturnExpr::can_cast(kind)
            || ast::PrefixExpr::can_cast(kind)
            || ast::FormatArgsArg::can_cast(kind)
            || ast::RecordExprField::can_cast(kind)
            || ast::BinExpr::cast(node.clone())
                .and_then(|expr| expr.rhs())
                .is_some_and(|expr| expr.syntax() == it)
            || ast::IndexExpr::cast(node)
                .and_then(|expr| expr.index())
                .is_some_and(|expr| expr.syntax() == it)
    };

    // We do not want to generate path completions when we are sandwiched between an item decl signature and its body.
    // ex. trait Foo $0 {}
    // in these cases parser recovery usually kicks in for our inserted identifier, causing it
    // to either be parsed as an ExprStmt or a ItemRecovery, depending on whether it is in a block
    // expression or an item list.
    // The following code checks if the body is missing, if it is we either cut off the body
    // from the item or it was missing in the first place
    let inbetween_body_and_decl_check = |node: SyntaxNode| {
        if let Some(NodeOrToken::Node(n)) =
            syntax::algo::non_trivia_sibling(node.into(), syntax::Direction::Prev)
            && let Some(item) = ast::Item::cast(n)
        {
            let is_inbetween = match &item {
                ast::Item::Const(it) => it.body().is_none() && it.semicolon_token().is_none(),
                ast::Item::Enum(it) => it.variant_list().is_none(),
                ast::Item::ExternBlock(it) => it.extern_item_list().is_none(),
                ast::Item::Fn(it) => it.body().is_none() && it.semicolon_token().is_none(),
                ast::Item::Impl(it) => it.assoc_item_list().is_none(),
                ast::Item::Module(it) => it.item_list().is_none() && it.semicolon_token().is_none(),
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
        None
    };

    let generic_arg_location = |arg: ast::GenericArg| {
        let mut override_location = None;
        let location = find_opt_node_in_file_compensated(
            sema,
            original_file,
            arg.syntax().parent().and_then(ast::GenericArgList::cast),
        )
        .map(|args| {
            let mut in_trait = None;
            let param = (|| {
                let parent = args.syntax().parent()?;
                let params = match_ast! {
                    match parent {
                        ast::PathSegment(segment) => {
                            match sema.resolve_path(&segment.parent_path().top_path())? {
                                hir::PathResolution::Def(def) => match def {
                                    hir::ModuleDef::Function(func) => {
                                         sema.source(func)?.value.generic_param_list()
                                    }
                                    hir::ModuleDef::Adt(adt) => {
                                        sema.source(adt)?.value.generic_param_list()
                                    }
                                    hir::ModuleDef::Variant(variant) => {
                                        sema.source(variant.parent_enum(sema.db))?.value.generic_param_list()
                                    }
                                    hir::ModuleDef::Trait(trait_) => {
                                        if let ast::GenericArg::AssocTypeArg(arg) = &arg {
                                            let arg_name = arg.name_ref()?;
                                            let arg_name = arg_name.text();
                                            for item in trait_.items_with_supertraits(sema.db) {
                                                match item {
                                                    hir::AssocItem::TypeAlias(assoc_ty) => {
                                                        if assoc_ty.name(sema.db).as_str() == arg_name {
                                                            override_location = Some(TypeLocation::AssocTypeEq);
                                                            return None;
                                                        }
                                                    },
                                                    hir::AssocItem::Const(const_) => {
                                                        if const_.name(sema.db)?.as_str() == arg_name {
                                                            override_location =  Some(TypeLocation::AssocConstEq);
                                                            return None;
                                                        }
                                                    },
                                                    _ => (),
                                                }
                                            }
                                            return None;
                                        } else {
                                            in_trait = Some(trait_);
                                            sema.source(trait_)?.value.generic_param_list()
                                        }
                                    }
                                    hir::ModuleDef::TypeAlias(ty_) => {
                                        sema.source(ty_)?.value.generic_param_list()
                                    }
                                    _ => None,
                                },
                                _ => None,
                            }
                        },
                        ast::MethodCallExpr(call) => {
                            let func = sema.resolve_method_call(&call)?;
                            sema.source(func)?.value.generic_param_list()
                        },
                        ast::AssocTypeArg(arg) => {
                            let trait_ = ast::PathSegment::cast(arg.syntax().parent()?.parent()?)?;
                            match sema.resolve_path(&trait_.parent_path().top_path())? {
                                hir::PathResolution::Def(hir::ModuleDef::Trait(trait_)) =>  {
                                        let arg_name = arg.name_ref()?;
                                        let arg_name = arg_name.text();
                                        let trait_items = trait_.items_with_supertraits(sema.db);
                                        let assoc_ty = trait_items.iter().find_map(|item| match item {
                                            hir::AssocItem::TypeAlias(assoc_ty) => {
                                                (assoc_ty.name(sema.db).as_str() == arg_name)
                                                    .then_some(assoc_ty)
                                            },
                                            _ => None,
                                        })?;
                                        sema.source(*assoc_ty)?.value.generic_param_list()
                                    }
                                _ => None,
                            }
                        },
                        _ => None,
                    }
                }?;
                // Determine the index of the argument in the `GenericArgList` and match it with
                // the corresponding parameter in the `GenericParamList`. Since lifetime parameters
                // are often omitted, ignore them for the purposes of matching the argument with
                // its parameter unless a lifetime argument is provided explicitly. That is, for
                // `struct S<'a, 'b, T>`, match `S::<$0>` to `T` and `S::<'a, $0, _>` to `'b`.
                // FIXME: This operates on the syntax tree and will produce incorrect results when
                // generic parameters are disabled by `#[cfg]` directives. It should operate on the
                // HIR, but the functionality necessary to do so is not exposed at the moment.
                let mut explicit_lifetime_arg = false;
                let arg_idx = arg
                    .syntax()
                    .siblings(Direction::Prev)
                    // Skip the node itself
                    .skip(1)
                    .map(|arg| if ast::LifetimeArg::can_cast(arg.kind()) { explicit_lifetime_arg = true })
                    .count();
                let param_idx = if explicit_lifetime_arg {
                    arg_idx
                } else {
                    // Lifetimes parameters always precede type and generic parameters,
                    // so offset the argument index by the total number of lifetime params
                    arg_idx + params.lifetime_params().count()
                };
                params.generic_params().nth(param_idx)
            })();
            (args, in_trait, param)
        });
        let (arg_list, of_trait, corresponding_param) = match location {
            Some((arg_list, of_trait, param)) => (Some(arg_list), of_trait, param),
            _ => (None, None, None),
        };
        override_location.unwrap_or(TypeLocation::GenericArg {
            args: arg_list,
            of_trait,
            corresponding_param,
        })
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
                ast::Static(it) => {
                    let name = find_opt_node_in_file(original_file, it.name())?;
                    let original = ast::Static::cast(name.syntax().parent()?)?;
                    TypeLocation::TypeAscription(TypeAscriptionTarget::Const(original.body()))
                },
                ast::RetType(it) => {
                    it.thin_arrow_token()?;
                    let parent = match ast::Fn::cast(parent.parent()?) {
                        Some(it) => it.param_list(),
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
                    it.colon_token()?;
                    TypeLocation::TypeAscription(TypeAscriptionTarget::FnParam(find_opt_node_in_file(original_file, it.pat())))
                },
                ast::LetStmt(it) => {
                    it.colon_token()?;
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
                ast::GenericArg(it) => generic_arg_location(it),
                // is this case needed?
                ast::GenericArgList(it) => {
                    let args = find_opt_node_in_file_compensated(sema, original_file, Some(it));
                    TypeLocation::GenericArg { args, of_trait: None, corresponding_param: None }
                },
                ast::TupleField(_) => TypeLocation::TupleField,
                _ => return None,
            }
        };
        Some(res)
    };

    let make_path_kind_expr = |expr: ast::Expr| {
        let it = expr.syntax();
        let prev_token = iter::successors(it.first_token(), |it| it.prev_token())
            .skip(1)
            .find(|it| !it.kind().is_trivia());
        let in_block_expr = is_in_block(it);
        let (in_loop_body, innermost_breakable) = is_in_breakable(it).unzip();
        let after_if_expr = is_after_if_expr(it.clone());
        let after_amp = prev_token.as_ref().is_some_and(|it| it.kind() == SyntaxKind::AMP);
        let ref_expr_parent = prev_token.and_then(|it| it.parent()).and_then(ast::RefExpr::cast);
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
            let fn_self_param =
                |fn_: ast::Fn| sema.to_def(&fn_).and_then(|it| it.self_param(sema.db));
            let closure_this_param = |closure: ast::ClosureExpr| {
                if closure.param_list()?.params().next()?.pat()?.syntax().text() != "this" {
                    return None;
                }
                sema.type_of_expr(&closure.into())
                    .and_then(|it| it.original.as_callable(sema.db))
                    .and_then(|it| it.params().into_iter().next())
            };
            let find_fn_self_param = |it: SyntaxNode| {
                match_ast! {
                    match it {
                        ast::Fn(fn_) => Some(fn_self_param(fn_).map(Either::Left)),
                        ast::ClosureExpr(f) => closure_this_param(f).map(Either::Right).map(Some),
                        ast::MacroCall(_) => None,
                        ast::Item(_) => Some(None),
                        _ => None,
                    }
                }
            };

            match find_node_in_file_compensated(sema, original_file, &expr) {
                Some(it) => {
                    // buggy
                    let innermost_ret_ty = sema
                        .ancestors_with_macros(it.syntax().clone())
                        .find_map(find_ret_ty)
                        .flatten();

                    let self_param = sema
                        .ancestors_with_macros(it.syntax().clone())
                        .find_map(find_fn_self_param)
                        .flatten();
                    (innermost_ret_ty, self_param)
                }
                None => (None, None),
            }
        };
        let innermost_breakable_ty = innermost_breakable
            .and_then(ast::Expr::cast)
            .and_then(|expr| find_node_in_file_compensated(sema, original_file, &expr))
            .and_then(|expr| sema.type_of_expr(&expr))
            .map(|ty| if ty.original.is_never() { ty.adjusted() } else { ty.original() });
        let is_func_update = func_update_record(it);
        let in_condition = is_in_condition(&expr);
        let after_incomplete_let = after_incomplete_let(it.clone()).is_some();
        let incomplete_expr_stmt =
            it.parent().and_then(ast::ExprStmt::cast).map(|it| it.semicolon_token().is_none());
        let before_else_kw = before_else_kw(it);
        let incomplete_let = left_ancestors(it.parent())
            .find_map(ast::LetStmt::cast)
            .is_some_and(|it| it.semicolon_token().is_none())
            || after_incomplete_let && incomplete_expr_stmt.unwrap_or(true) && !before_else_kw;
        let in_value = is_in_value(it);
        let impl_ = fetch_immediate_impl_or_trait(sema, original_file, expr.syntax())
            .and_then(Either::left);

        let in_match_guard = match it.parent().and_then(ast::MatchArm::cast) {
            Some(arm) => arm
                .fat_arrow_token()
                .is_none_or(|arrow| it.text_range().start() < arrow.text_range().start()),
            None => false,
        };

        PathKind::Expr {
            expr_ctx: PathExprCtx {
                in_block_expr,
                in_breakable: in_loop_body,
                after_if_expr,
                before_else_kw,
                in_condition,
                ref_expr_parent,
                after_amp,
                is_func_update,
                innermost_ret_ty,
                innermost_breakable_ty,
                self_param,
                in_value,
                incomplete_let,
                after_incomplete_let,
                impl_,
                in_match_guard,
            },
        }
    };
    let make_path_kind_type = |ty: ast::Type| {
        let location = type_location(ty.syntax());
        PathKind::Type { location: location.unwrap_or(TypeLocation::Other) }
    };

    let kind_item = |it: &SyntaxNode| {
        let parent = it.parent()?;
        let kind = match_ast! {
            match parent {
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
                ast::ExternItemList(it) => {
                    let exn_blk = it.syntax().parent().and_then(ast::ExternBlock::cast);
                    PathKind::Item {
                        kind: ItemListKind::ExternBlock {
                            is_unsafe: exn_blk.and_then(|it| it.unsafe_token()).is_some(),
                        }
                    }
                },
                ast::SourceFile(_) => PathKind::Item { kind: ItemListKind::SourceFile },
                _ => return None,
            }
        };
        Some(kind)
    };

    let mut kind_macro_call = |it: ast::MacroCall| {
        path_ctx.has_macro_bang = it.excl_token().is_some();
        let parent = it.syntax().parent()?;
        if let Some(kind) = kind_item(it.syntax()) {
            return Some(kind);
        }
        let kind = match_ast! {
            match parent {
                ast::MacroExpr(expr) => make_path_kind_expr(expr.into()),
                ast::MacroPat(it) => PathKind::Pat { pat_ctx: pattern_context_for(sema, original_file, it.into())},
                ast::MacroType(ty) => make_path_kind_type(ty.into()),
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
        let derive_helpers = annotated_item_kind
            .filter(|kind| {
                matches!(
                    kind,
                    SyntaxKind::STRUCT
                        | SyntaxKind::ENUM
                        | SyntaxKind::UNION
                        | SyntaxKind::VARIANT
                        | SyntaxKind::TUPLE_FIELD
                        | SyntaxKind::RECORD_FIELD
                )
            })
            .and_then(|_| nameref.as_ref()?.syntax().ancestors().find_map(ast::Adt::cast))
            .and_then(|adt| sema.derive_helpers_in_scope(&adt))
            .unwrap_or_default();
        Some(PathKind::Attr { attr_ctx: AttrCtx { kind, annotated_item_kind, derive_helpers } })
    };

    // Infer the path kind
    let parent = path.syntax().parent()?;
    let kind = 'find_kind: {
        if parent.kind() == SyntaxKind::ERROR {
            if let Some(kind) = inbetween_body_and_decl_check(parent.clone()) {
                return Some(make_res(NameRefKind::Keyword(kind)));
            }

            break 'find_kind kind_item(&parent)?;
        }
        match_ast! {
            match parent {
                ast::PathType(it) => make_path_kind_type(it.into()),
                ast::PathExpr(it) => {
                    if let Some(p) = it.syntax().parent() {
                        let p_kind = p.kind();
                        // The syntax node of interest, for which we want to check whether
                        // it is sandwiched between an item decl signature and its body.
                        let probe = if ast::ExprStmt::can_cast(p_kind) {
                            Some(p)
                        } else if ast::StmtList::can_cast(p_kind) {
                            Some(it.syntax().clone())
                        } else {
                            None
                        };
                        if let Some(kind) = probe.and_then(inbetween_body_and_decl_check) {
                            return Some(make_res(NameRefKind::Keyword(kind)));
                        }
                    }

                    path_ctx.has_call_parens = it.syntax().parent().is_some_and(|it| ast::CallExpr::cast(it).is_some_and(|it| has_parens(&it)));

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
                // completing inside a qualifier
                ast::Path(parent) => {
                    path_ctx.parent = Some(parent.clone());
                    let parent = iter::successors(Some(parent), |it| it.parent_path()).last()?.syntax().parent()?;
                    match_ast! {
                        match parent {
                            ast::PathType(it) => make_path_kind_type(it.into()),
                            ast::PathExpr(it) => {
                                path_ctx.has_call_parens = it.syntax().parent().is_some_and(|it| ast::CallExpr::cast(it).is_some_and(|it| has_parens(&it)));

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
                ast::RecordExpr(it) => {
                    // A record expression in this position is usually a result of parsing recovery, so check that
                    if let Some(kind) = inbetween_body_and_decl_check(it.syntax().clone()) {
                        return Some(make_res(NameRefKind::Keyword(kind)));
                    }
                    make_path_kind_expr(it.into())
                },
                _ => return None,
            }
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
    } else if let Some(segment) = path.segment()
        && segment.coloncolon_token().is_some()
    {
        path_ctx.qualified = Qualified::Absolute;
    }

    let mut qualifier_ctx = QualifierCtx::default();
    if path_ctx.is_trivial_path() {
        // fetch the full expression that may have qualifiers attached to it
        let top_node = match path_ctx.kind {
            PathKind::Expr { expr_ctx: PathExprCtx { in_block_expr: true, .. } } => {
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
            PathKind::Item { .. } => parent.ancestors().find(|it| it.kind() == SyntaxKind::ERROR),
            _ => None,
        };
        if let Some(top) = top_node {
            if let Some(NodeOrToken::Node(error_node)) =
                syntax::algo::non_trivia_sibling(top.clone().into(), syntax::Direction::Prev)
                && error_node.kind() == SyntaxKind::ERROR
            {
                for token in error_node.children_with_tokens().filter_map(NodeOrToken::into_token) {
                    match token.kind() {
                        SyntaxKind::UNSAFE_KW => qualifier_ctx.unsafe_tok = Some(token),
                        SyntaxKind::ASYNC_KW => qualifier_ctx.async_tok = Some(token),
                        SyntaxKind::SAFE_KW => qualifier_ctx.safe_tok = Some(token),
                        _ => {}
                    }
                }
                qualifier_ctx.vis_node = error_node.children().find_map(ast::Visibility::cast);
                qualifier_ctx.abi_node = error_node.children().find_map(ast::Abi::cast);
            }

            if let PathKind::Item { .. } = path_ctx.kind
                && qualifier_ctx.none()
                && let Some(t) = top.first_token()
                && let Some(prev) =
                    t.prev_token().and_then(|t| syntax::algo::skip_trivia_token(t, Direction::Prev))
                && ![T![;], T!['}'], T!['{'], T![']']].contains(&prev.kind())
            {
                // This was inferred to be an item position path, but it seems
                // to be part of some other broken node which leaked into an item
                // list
                return None;
            }
        }
    }
    Some((NameRefContext { nameref, kind: NameRefKind::Path(path_ctx) }, qualifier_ctx))
}

/// When writing in the middle of some code the following situation commonly occurs (`|` denotes the cursor):
/// ```ignore
/// value.method|
/// (1, 2, 3)
/// ```
/// Here, we want to complete the method parentheses & arguments (if the corresponding settings are on),
/// but the thing is parsed as a method call with parentheses. Therefore we use heuristics: if the parentheses
/// are on the next line, consider them non-existent.
fn has_parens(node: &dyn HasArgList) -> bool {
    let Some(arg_list) = node.arg_list() else { return false };
    if arg_list.l_paren_token().is_none() {
        return false;
    }
    let prev_siblings = iter::successors(arg_list.syntax().prev_sibling_or_token(), |it| {
        it.prev_sibling_or_token()
    });
    prev_siblings
        .take_while(|syntax| syntax.kind().is_trivia())
        .filter_map(|syntax| {
            syntax.into_token().filter(|token| token.kind() == SyntaxKind::WHITESPACE)
        })
        .all(|whitespace| !whitespace.text().contains('\n'))
}

fn pattern_context_for(
    sema: &Semantics<'_, RootDatabase>,
    original_file: &SyntaxNode,
    pat: ast::Pat,
) -> PatternContext {
    let mut param_ctx = None;

    let mut missing_variants = vec![];
    let is_pat_like = |kind| {
        ast::Pat::can_cast(kind)
            || ast::RecordPatField::can_cast(kind)
            || ast::RecordPatFieldList::can_cast(kind)
    };

    let (refutability, has_type_ascription) = pat
        .syntax()
        .ancestors()
        .find(|it| !is_pat_like(it.kind()))
        .map_or((PatternRefutability::Irrefutable, false), |node| {
            let refutability = match_ast! {
                match node {
                    ast::LetStmt(let_) => return (PatternRefutability::Refutable, let_.ty().is_some()),
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
                                    let expr_opt = find_opt_node_in_file(original_file, match_expr.expr());

                                    expr_opt.and_then(|expr| {
                                        sema.type_of_expr(&expr)?
                                        .adjusted()
                                        .autoderef(sema.db)
                                        .find_map(|ty| match ty.as_adt() {
                                            Some(hir::Adt::Enum(e)) => Some(e),
                                            _ => None,
                                        }).map(|enum_| enum_.variants(sema.db))
                                    })
                                }).map(|variants| variants.iter().filter_map(|variant| {
                                        let variant_name = variant.name(sema.db);

                                        let variant_already_present = match_arm_list.arms().any(|arm| {
                                            arm.pat().and_then(|pat| {
                                                let pat_already_present = pat.syntax().to_string().contains(variant_name.as_str());
                                                pat_already_present.then_some(pat_already_present)
                                            }).is_some()
                                        });

                                        (!variant_already_present).then_some(*variant)
                                    }).collect::<Vec<Variant>>())
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

    // Only suggest name in let-stmt or fn param
    let should_suggest_name = matches!(
            &pat,
            ast::Pat::IdentPat(it)
                if it.syntax()
                .parent().is_some_and(|node| {
                    let kind = node.kind();
                    ast::LetStmt::can_cast(kind) || ast::Param::can_cast(kind)
                })
    );

    PatternContext {
        refutability,
        param_ctx,
        has_type_ascription,
        should_suggest_name,
        after_if_expr: is_after_if_expr(pat.syntax().clone()),
        parent_pat: pat.syntax().parent().and_then(ast::Pat::cast),
        mut_token,
        ref_token,
        record_pat: None,
        impl_or_trait: fetch_immediate_impl_or_trait(sema, original_file, pat.syntax()),
        missing_variants,
    }
}

fn fetch_immediate_impl_or_trait(
    sema: &Semantics<'_, RootDatabase>,
    original_file: &SyntaxNode,
    node: &SyntaxNode,
) -> Option<Either<ast::Impl, ast::Trait>> {
    let mut ancestors = ancestors_in_file_compensated(sema, original_file, node)?
        .filter_map(ast::Item::cast)
        .filter(|it| !matches!(it, ast::Item::MacroCall(_)));

    match ancestors.next()? {
        ast::Item::Const(_) | ast::Item::Fn(_) | ast::Item::TypeAlias(_) => (),
        ast::Item::Impl(it) => return Some(Either::Left(it)),
        ast::Item::Trait(it) => return Some(Either::Right(it)),
        _ => return None,
    }
    match ancestors.next()? {
        ast::Item::Impl(it) => Some(Either::Left(it)),
        ast::Item::Trait(it) => Some(Either::Right(it)),
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

fn left_ancestors(node: Option<SyntaxNode>) -> impl Iterator<Item = SyntaxNode> {
    node.into_iter().flat_map(|node| {
        let end = node.text_range().end();
        node.ancestors().take_while(move |it| it.text_range().end() == end)
    })
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

fn is_in_breakable(node: &SyntaxNode) -> Option<(BreakableKind, SyntaxNode)> {
    node.ancestors()
        .take_while(|it| it.kind() != SyntaxKind::FN && it.kind() != SyntaxKind::CLOSURE_EXPR)
        .find_map(|it| {
            let (breakable, loop_body) = match_ast! {
                match it {
                    ast::ForExpr(it) => (BreakableKind::For, it.loop_body()?),
                    ast::WhileExpr(it) => (BreakableKind::While, it.loop_body()?),
                    ast::LoopExpr(it) => (BreakableKind::Loop, it.loop_body()?),
                    ast::BlockExpr(it) => return it.label().map(|_| (BreakableKind::Block, it.syntax().clone())),
                    _ => return None,
                }
            };
            loop_body.syntax().text_range().contains_range(node.text_range())
                .then_some((breakable, it))
        })
}

fn is_in_block(node: &SyntaxNode) -> bool {
    if has_in_newline_expr_first(node) {
        return true;
    };
    node.parent()
        .map(|node| ast::ExprStmt::can_cast(node.kind()) || ast::StmtList::can_cast(node.kind()))
        .unwrap_or(false)
}

/// Similar to `has_parens`, heuristic sensing incomplete statement before ambiguous `Expr`
///
/// Heuristic:
///
/// If the `PathExpr` is left part of the `Expr` and there is a newline after the `PathExpr`,
/// it is considered that the `PathExpr` is not part of the `Expr`.
fn has_in_newline_expr_first(node: &SyntaxNode) -> bool {
    if ast::PathExpr::can_cast(node.kind())
        && let Some(NodeOrToken::Token(next)) = node.next_sibling_or_token()
        && next.kind() == SyntaxKind::WHITESPACE
        && next.text().contains('\n')
        && let Some(stmt_like) = node
            .ancestors()
            .take_while(|it| it.text_range().start() == node.text_range().start())
            .filter_map(Either::<ast::ExprStmt, ast::Expr>::cast)
            .last()
    {
        stmt_like.syntax().parent().and_then(ast::StmtList::cast).is_some()
    } else {
        false
    }
}

fn is_after_if_expr(node: SyntaxNode) -> bool {
    let node = match node.parent().and_then(Either::<ast::ExprStmt, ast::MatchArm>::cast) {
        Some(stmt) => stmt.syntax().clone(),
        None => node,
    };
    let Some(prev_token) = previous_non_trivia_token(node) else { return false };
    prev_token
        .parent_ancestors()
        .take_while(|it| it.text_range().end() == prev_token.text_range().end())
        .find_map(ast::IfExpr::cast)
        .is_some()
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

fn prev_special_biased_token_at_trivia(mut token: SyntaxToken) -> SyntaxToken {
    while token.kind().is_trivia()
        && let Some(prev) = token.prev_token()
        && let T![=]
        | T![+=]
        | T![/=]
        | T![*=]
        | T![%=]
        | T![>>=]
        | T![<<=]
        | T![-=]
        | T![|=]
        | T![&=]
        | T![^=]
        | T![|]
        | T![return]
        | T![break]
        | T![continue]
        | T![lifetime_ident] = prev.kind()
    {
        token = prev
    }
    token
}
