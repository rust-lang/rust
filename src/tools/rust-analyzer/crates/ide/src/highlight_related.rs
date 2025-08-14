use std::iter;

use hir::{EditionedFileId, FilePosition, FileRange, HirFileId, InFile, Semantics, db};
use ide_db::{
    FxHashMap, FxHashSet, RootDatabase,
    defs::{Definition, IdentClass},
    helpers::pick_best_token,
    search::{FileReference, ReferenceCategory, SearchScope},
    syntax_helpers::node_ext::{
        eq_label_lt, for_each_tail_expr, full_path_of_name_ref, is_closure_or_blk_with_modif,
        preorder_expr_with_ctx_checker,
    },
};
use syntax::{
    AstNode,
    SyntaxKind::{self, IDENT, INT_NUMBER},
    SyntaxToken, T, TextRange, WalkEvent,
    ast::{self, HasLoopBody},
    match_ast,
};

use crate::{NavigationTarget, TryToNav, goto_definition, navigation_target::ToNav};

#[derive(PartialEq, Eq, Hash)]
pub struct HighlightedRange {
    pub range: TextRange,
    // FIXME: This needs to be more precise. Reference category makes sense only
    // for references, but we also have defs. And things like exit points are
    // neither.
    pub category: ReferenceCategory,
}

#[derive(Default, Clone)]
pub struct HighlightRelatedConfig {
    pub references: bool,
    pub exit_points: bool,
    pub break_points: bool,
    pub closure_captures: bool,
    pub yield_points: bool,
    pub branch_exit_points: bool,
}

type HighlightMap = FxHashMap<EditionedFileId, FxHashSet<HighlightedRange>>;

// Feature: Highlight Related
//
// Highlights constructs related to the thing under the cursor:
//
// 1. if on an identifier, highlights all references to that identifier in the current file
//      * additionally, if the identifier is a trait in a where clause, type parameter trait bound or use item, highlights all references to that trait's assoc items in the corresponding scope
// 1. if on an `async` or `await` token, highlights all yield points for that async context
// 1. if on a `return` or `fn` keyword, `?` character or `->` return type arrow, highlights all exit points for that context
// 1. if on a `break`, `loop`, `while` or `for` token, highlights all break points for that loop or block context
// 1. if on a `move` or `|` token that belongs to a closure, highlights all captures of the closure.
//
// Note: `?`, `|` and `->` do not currently trigger this behavior in the VSCode editor.
pub(crate) fn highlight_related(
    sema: &Semantics<'_, RootDatabase>,
    config: HighlightRelatedConfig,
    ide_db::FilePosition { offset, file_id }: ide_db::FilePosition,
) -> Option<Vec<HighlightedRange>> {
    let _p = tracing::info_span!("highlight_related").entered();
    let file_id = sema
        .attach_first_edition(file_id)
        .unwrap_or_else(|| EditionedFileId::current_edition(sema.db, file_id));
    let syntax = sema.parse(file_id).syntax().clone();

    let token = pick_best_token(syntax.token_at_offset(offset), |kind| match kind {
        T![?] => 4, // prefer `?` when the cursor is sandwiched like in `await$0?`
        T![->] | T![=>] => 4,
        kind if kind.is_keyword(file_id.edition(sema.db)) => 3,
        IDENT | INT_NUMBER => 2,
        T![|] => 1,
        _ => 0,
    })?;
    // most if not all of these should be re-implemented with information seeded from hir
    match token.kind() {
        T![?] if config.exit_points && token.parent().and_then(ast::TryExpr::cast).is_some() => {
            highlight_exit_points(sema, token).remove(&file_id)
        }
        T![fn] | T![return] | T![->] if config.exit_points => {
            highlight_exit_points(sema, token).remove(&file_id)
        }
        T![match] | T![=>] | T![if] if config.branch_exit_points => {
            highlight_branch_exit_points(sema, token).remove(&file_id)
        }
        T![await] | T![async] if config.yield_points => {
            highlight_yield_points(sema, token).remove(&file_id)
        }
        T![for] if config.break_points && token.parent().and_then(ast::ForExpr::cast).is_some() => {
            highlight_break_points(sema, token).remove(&file_id)
        }
        T![break] | T![loop] | T![while] | T![continue] if config.break_points => {
            highlight_break_points(sema, token).remove(&file_id)
        }
        T![unsafe] if token.parent().and_then(ast::BlockExpr::cast).is_some() => {
            highlight_unsafe_points(sema, token).remove(&file_id)
        }
        T![|] if config.closure_captures => highlight_closure_captures(sema, token, file_id),
        T![move] if config.closure_captures => highlight_closure_captures(sema, token, file_id),
        _ if config.references => {
            highlight_references(sema, token, FilePosition { file_id, offset })
        }
        _ => None,
    }
}

fn highlight_closure_captures(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
    file_id: EditionedFileId,
) -> Option<Vec<HighlightedRange>> {
    let closure = token.parent_ancestors().take(2).find_map(ast::ClosureExpr::cast)?;
    let search_range = closure.body()?.syntax().text_range();
    let ty = &sema.type_of_expr(&closure.into())?.original;
    let c = ty.as_closure()?;
    Some(
        c.captured_items(sema.db)
            .into_iter()
            .map(|capture| capture.local())
            .flat_map(|local| {
                let usages = Definition::Local(local)
                    .usages(sema)
                    .in_scope(&SearchScope::file_range(FileRange { file_id, range: search_range }))
                    .include_self_refs()
                    .all()
                    .references
                    .remove(&file_id)
                    .into_iter()
                    .flatten()
                    .map(|FileReference { category, range, .. }| HighlightedRange {
                        range,
                        category,
                    });
                let category = if local.is_mut(sema.db) {
                    ReferenceCategory::WRITE
                } else {
                    ReferenceCategory::empty()
                };
                local
                    .sources(sema.db)
                    .into_iter()
                    .flat_map(|x| x.to_nav(sema.db))
                    .filter(|decl| decl.file_id == file_id.file_id(sema.db))
                    .filter_map(|decl| decl.focus_range)
                    .map(move |range| HighlightedRange { range, category })
                    .chain(usages)
            })
            .collect(),
    )
}

fn highlight_references(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
    FilePosition { file_id, offset }: FilePosition,
) -> Option<Vec<HighlightedRange>> {
    let defs = if let Some((range, _, _, resolution)) =
        sema.check_for_format_args_template(token.clone(), offset)
    {
        match resolution.map(Definition::from) {
            Some(def) => iter::once(def).collect(),
            None => {
                return Some(vec![HighlightedRange {
                    range,
                    category: ReferenceCategory::empty(),
                }]);
            }
        }
    } else {
        find_defs(sema, token.clone())
    };
    let usages = defs
        .iter()
        .filter_map(|&d| {
            d.usages(sema)
                .in_scope(&SearchScope::single_file(file_id))
                .include_self_refs()
                .all()
                .references
                .remove(&file_id)
        })
        .flatten()
        .map(|FileReference { category, range, .. }| HighlightedRange { range, category });
    let mut res = FxHashSet::default();
    for &def in &defs {
        // highlight trait usages
        if let Definition::Trait(t) = def {
            let trait_item_use_scope = (|| {
                let name_ref = token.parent().and_then(ast::NameRef::cast)?;
                let path = full_path_of_name_ref(&name_ref)?;
                let parent = path.syntax().parent()?;
                match_ast! {
                    match parent {
                        ast::UseTree(it) => it.syntax().ancestors().find(|it| {
                            ast::SourceFile::can_cast(it.kind()) || ast::Module::can_cast(it.kind())
                        }).zip(Some(true)),
                        ast::PathType(it) => it
                            .syntax()
                            .ancestors()
                            .nth(2)
                            .and_then(ast::TypeBoundList::cast)?
                            .syntax()
                            .parent()
                            .filter(|it| ast::WhereClause::can_cast(it.kind()) || ast::TypeParam::can_cast(it.kind()))?
                            .ancestors()
                            .find(|it| {
                                ast::Item::can_cast(it.kind())
                            }).zip(Some(false)),
                        _ => None,
                    }
                }
            })();
            if let Some((trait_item_use_scope, use_tree)) = trait_item_use_scope {
                res.extend(
                    if use_tree { t.items(sema.db) } else { t.items_with_supertraits(sema.db) }
                        .into_iter()
                        .filter_map(|item| {
                            Definition::from(item)
                                .usages(sema)
                                .set_scope(Some(&SearchScope::file_range(FileRange {
                                    file_id,
                                    range: trait_item_use_scope.text_range(),
                                })))
                                .include_self_refs()
                                .all()
                                .references
                                .remove(&file_id)
                        })
                        .flatten()
                        .map(|FileReference { category, range, .. }| HighlightedRange {
                            range,
                            category,
                        }),
                );
            }
        }

        // highlight the tail expr of the labelled block
        if matches!(def, Definition::Label(_)) {
            let label = token.parent_ancestors().nth(1).and_then(ast::Label::cast);
            if let Some(block) =
                label.and_then(|label| label.syntax().parent()).and_then(ast::BlockExpr::cast)
            {
                for_each_tail_expr(&block.into(), &mut |tail| {
                    if !matches!(tail, ast::Expr::BreakExpr(_)) {
                        res.insert(HighlightedRange {
                            range: tail.syntax().text_range(),
                            category: ReferenceCategory::empty(),
                        });
                    }
                });
            }
        }

        // highlight the defs themselves
        match def {
            Definition::Local(local) => {
                let category = if local.is_mut(sema.db) {
                    ReferenceCategory::WRITE
                } else {
                    ReferenceCategory::empty()
                };
                local
                    .sources(sema.db)
                    .into_iter()
                    .flat_map(|x| x.to_nav(sema.db))
                    .filter(|decl| decl.file_id == file_id.file_id(sema.db))
                    .filter_map(|decl| decl.focus_range)
                    .map(|range| HighlightedRange { range, category })
                    .for_each(|x| {
                        res.insert(x);
                    });
            }
            def => {
                let navs = match def {
                    Definition::Module(module) => {
                        NavigationTarget::from_module_to_decl(sema.db, module)
                    }
                    def => match def.try_to_nav(sema.db) {
                        Some(it) => it,
                        None => continue,
                    },
                };
                for nav in navs {
                    if nav.file_id != file_id.file_id(sema.db) {
                        continue;
                    }
                    let hl_range = nav.focus_range.map(|range| {
                        let category = if matches!(def, Definition::Local(l) if l.is_mut(sema.db)) {
                            ReferenceCategory::WRITE
                        } else {
                            ReferenceCategory::empty()
                        };
                        HighlightedRange { range, category }
                    });
                    if let Some(hl_range) = hl_range {
                        res.insert(hl_range);
                    }
                }
            }
        }
    }

    res.extend(usages);
    if res.is_empty() { None } else { Some(res.into_iter().collect()) }
}

pub(crate) fn highlight_branch_exit_points(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
) -> FxHashMap<EditionedFileId, Vec<HighlightedRange>> {
    let mut highlights: HighlightMap = FxHashMap::default();

    let push_to_highlights = |file_id, range, highlights: &mut HighlightMap| {
        if let Some(FileRange { file_id, range }) = original_frange(sema.db, file_id, range) {
            let hrange = HighlightedRange { category: ReferenceCategory::empty(), range };
            highlights.entry(file_id).or_default().insert(hrange);
        }
    };

    let push_tail_expr = |tail: Option<ast::Expr>, highlights: &mut HighlightMap| {
        let Some(tail) = tail else {
            return;
        };

        for_each_tail_expr(&tail, &mut |tail| {
            let file_id = sema.hir_file_for(tail.syntax());
            let range = tail.syntax().text_range();
            push_to_highlights(file_id, Some(range), highlights);
        });
    };

    let nodes = goto_definition::find_branch_root(sema, &token).into_iter();
    match token.kind() {
        T![match] => {
            for match_expr in nodes.filter_map(ast::MatchExpr::cast) {
                let file_id = sema.hir_file_for(match_expr.syntax());
                let range = match_expr.match_token().map(|token| token.text_range());
                push_to_highlights(file_id, range, &mut highlights);

                let Some(arm_list) = match_expr.match_arm_list() else {
                    continue;
                };
                for arm in arm_list.arms() {
                    push_tail_expr(arm.expr(), &mut highlights);
                }
            }
        }
        T![=>] => {
            for arm in nodes.filter_map(ast::MatchArm::cast) {
                let file_id = sema.hir_file_for(arm.syntax());
                let range = arm.fat_arrow_token().map(|token| token.text_range());
                push_to_highlights(file_id, range, &mut highlights);

                push_tail_expr(arm.expr(), &mut highlights);
            }
        }
        T![if] => {
            for mut if_to_process in nodes.map(ast::IfExpr::cast) {
                while let Some(cur_if) = if_to_process.take() {
                    let file_id = sema.hir_file_for(cur_if.syntax());

                    let if_kw_range = cur_if.if_token().map(|token| token.text_range());
                    push_to_highlights(file_id, if_kw_range, &mut highlights);

                    if let Some(then_block) = cur_if.then_branch() {
                        push_tail_expr(Some(then_block.into()), &mut highlights);
                    }

                    match cur_if.else_branch() {
                        Some(ast::ElseBranch::Block(else_block)) => {
                            push_tail_expr(Some(else_block.into()), &mut highlights);
                            if_to_process = None;
                        }
                        Some(ast::ElseBranch::IfExpr(nested_if)) => if_to_process = Some(nested_if),
                        None => if_to_process = None,
                    }
                }
            }
        }
        _ => {}
    }

    highlights
        .into_iter()
        .map(|(file_id, ranges)| (file_id, ranges.into_iter().collect()))
        .collect()
}

fn hl_exit_points(
    sema: &Semantics<'_, RootDatabase>,
    def_token: Option<SyntaxToken>,
    body: ast::Expr,
) -> Option<HighlightMap> {
    let mut highlights: FxHashMap<EditionedFileId, FxHashSet<_>> = FxHashMap::default();

    let mut push_to_highlights = |file_id, range| {
        if let Some(FileRange { file_id, range }) = original_frange(sema.db, file_id, range) {
            let hrange = HighlightedRange { category: ReferenceCategory::empty(), range };
            highlights.entry(file_id).or_default().insert(hrange);
        }
    };

    if let Some(tok) = def_token {
        let file_id = sema.hir_file_for(&tok.parent()?);
        let range = Some(tok.text_range());
        push_to_highlights(file_id, range);
    }

    WalkExpandedExprCtx::new(sema).walk(&body, &mut |_, expr| {
        let file_id = sema.hir_file_for(expr.syntax());

        let range = match &expr {
            ast::Expr::TryExpr(try_) => try_.question_mark_token().map(|token| token.text_range()),
            ast::Expr::MethodCallExpr(_) | ast::Expr::CallExpr(_) | ast::Expr::MacroExpr(_)
                if sema.type_of_expr(&expr).is_some_and(|ty| ty.original.is_never()) =>
            {
                Some(expr.syntax().text_range())
            }
            _ => None,
        };

        push_to_highlights(file_id, range);
    });

    // We should handle `return` separately, because when it is used in a `try` block,
    // it will exit the outside function instead of the block itself.
    WalkExpandedExprCtx::new(sema)
        .with_check_ctx(&WalkExpandedExprCtx::is_async_const_block_or_closure)
        .walk(&body, &mut |_, expr| {
            let file_id = sema.hir_file_for(expr.syntax());

            let range = match &expr {
                ast::Expr::ReturnExpr(expr) => expr.return_token().map(|token| token.text_range()),
                _ => None,
            };

            push_to_highlights(file_id, range);
        });

    let tail = match body {
        ast::Expr::BlockExpr(b) => b.tail_expr(),
        e => Some(e),
    };

    if let Some(tail) = tail {
        for_each_tail_expr(&tail, &mut |tail| {
            let file_id = sema.hir_file_for(tail.syntax());
            let range = match tail {
                ast::Expr::BreakExpr(b) => b
                    .break_token()
                    .map_or_else(|| tail.syntax().text_range(), |tok| tok.text_range()),
                _ => tail.syntax().text_range(),
            };
            push_to_highlights(file_id, Some(range));
        });
    }
    Some(highlights)
}

// If `file_id` is None,
pub(crate) fn highlight_exit_points(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
) -> FxHashMap<EditionedFileId, Vec<HighlightedRange>> {
    let mut res = FxHashMap::default();
    for def in goto_definition::find_fn_or_blocks(sema, &token) {
        let new_map = match_ast! {
            match def {
                ast::Fn(fn_) => fn_.body().and_then(|body| hl_exit_points(sema, fn_.fn_token(), body.into())),
                ast::ClosureExpr(closure) => {
                    let pipe_tok = closure.param_list().and_then(|p| p.pipe_token());
                    closure.body().and_then(|body| hl_exit_points(sema, pipe_tok, body))
                },
                ast::BlockExpr(blk) => match blk.modifier() {
                    Some(ast::BlockModifier::Async(t)) => hl_exit_points(sema, Some(t), blk.into()),
                    Some(ast::BlockModifier::Try(t)) if token.kind() != T![return] => {
                        hl_exit_points(sema, Some(t), blk.into())
                    },
                    _ => continue,
                },
                _ => continue,
            }
        };
        merge_map(&mut res, new_map);
    }

    res.into_iter().map(|(file_id, ranges)| (file_id, ranges.into_iter().collect())).collect()
}

pub(crate) fn highlight_break_points(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
) -> FxHashMap<EditionedFileId, Vec<HighlightedRange>> {
    pub(crate) fn hl(
        sema: &Semantics<'_, RootDatabase>,
        cursor_token_kind: SyntaxKind,
        loop_token: Option<SyntaxToken>,
        label: Option<ast::Label>,
        expr: ast::Expr,
    ) -> Option<HighlightMap> {
        let mut highlights: FxHashMap<EditionedFileId, FxHashSet<_>> = FxHashMap::default();

        let mut push_to_highlights = |file_id, range| {
            if let Some(FileRange { file_id, range }) = original_frange(sema.db, file_id, range) {
                let hrange = HighlightedRange { category: ReferenceCategory::empty(), range };
                highlights.entry(file_id).or_default().insert(hrange);
            }
        };

        let label_lt = label.as_ref().and_then(|it| it.lifetime());

        if let Some(range) = cover_range(
            loop_token.as_ref().map(|tok| tok.text_range()),
            label.as_ref().map(|it| it.syntax().text_range()),
        ) {
            let file_id = loop_token
                .and_then(|tok| Some(sema.hir_file_for(&tok.parent()?)))
                .unwrap_or_else(|| sema.hir_file_for(label.unwrap().syntax()));
            push_to_highlights(file_id, Some(range));
        }

        WalkExpandedExprCtx::new(sema)
            .with_check_ctx(&WalkExpandedExprCtx::is_async_const_block_or_closure)
            .walk(&expr, &mut |depth, expr| {
                let file_id = sema.hir_file_for(expr.syntax());

                // Only highlight the `break`s for `break` and `continue`s for `continue`
                let (token, token_lt) = match expr {
                    ast::Expr::BreakExpr(b) if cursor_token_kind != T![continue] => {
                        (b.break_token(), b.lifetime())
                    }
                    ast::Expr::ContinueExpr(c) if cursor_token_kind != T![break] => {
                        (c.continue_token(), c.lifetime())
                    }
                    _ => return,
                };

                if !(depth == 1 && token_lt.is_none() || eq_label_lt(&label_lt, &token_lt)) {
                    return;
                }

                let text_range = cover_range(
                    token.map(|it| it.text_range()),
                    token_lt.map(|it| it.syntax().text_range()),
                );

                push_to_highlights(file_id, text_range);
            });

        if matches!(expr, ast::Expr::BlockExpr(_)) {
            for_each_tail_expr(&expr, &mut |tail| {
                if matches!(tail, ast::Expr::BreakExpr(_)) {
                    return;
                }

                let file_id = sema.hir_file_for(tail.syntax());
                let range = tail.syntax().text_range();
                push_to_highlights(file_id, Some(range));
            });
        }

        Some(highlights)
    }

    let Some(loops) = goto_definition::find_loops(sema, &token) else {
        return FxHashMap::default();
    };

    let mut res = FxHashMap::default();
    let token_kind = token.kind();
    for expr in loops {
        let new_map = match &expr {
            ast::Expr::LoopExpr(l) => hl(sema, token_kind, l.loop_token(), l.label(), expr),
            ast::Expr::ForExpr(f) => hl(sema, token_kind, f.for_token(), f.label(), expr),
            ast::Expr::WhileExpr(w) => hl(sema, token_kind, w.while_token(), w.label(), expr),
            ast::Expr::BlockExpr(e) => hl(sema, token_kind, None, e.label(), expr),
            _ => continue,
        };
        merge_map(&mut res, new_map);
    }

    res.into_iter().map(|(file_id, ranges)| (file_id, ranges.into_iter().collect())).collect()
}

pub(crate) fn highlight_yield_points(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
) -> FxHashMap<EditionedFileId, Vec<HighlightedRange>> {
    fn hl(
        sema: &Semantics<'_, RootDatabase>,
        async_token: Option<SyntaxToken>,
        body: Option<ast::Expr>,
    ) -> Option<HighlightMap> {
        let mut highlights: FxHashMap<EditionedFileId, FxHashSet<_>> = FxHashMap::default();

        let mut push_to_highlights = |file_id, range| {
            if let Some(FileRange { file_id, range }) = original_frange(sema.db, file_id, range) {
                let hrange = HighlightedRange { category: ReferenceCategory::empty(), range };
                highlights.entry(file_id).or_default().insert(hrange);
            }
        };

        let async_token = async_token?;
        let async_tok_file_id = sema.hir_file_for(&async_token.parent()?);
        push_to_highlights(async_tok_file_id, Some(async_token.text_range()));

        let Some(body) = body else {
            return Some(highlights);
        };

        WalkExpandedExprCtx::new(sema).walk(&body, &mut |_, expr| {
            let file_id = sema.hir_file_for(expr.syntax());

            let text_range = match expr {
                ast::Expr::AwaitExpr(expr) => expr.await_token(),
                ast::Expr::ReturnExpr(expr) => expr.return_token(),
                _ => None,
            }
            .map(|it| it.text_range());

            push_to_highlights(file_id, text_range);
        });

        Some(highlights)
    }

    let mut res = FxHashMap::default();
    for anc in goto_definition::find_fn_or_blocks(sema, &token) {
        let new_map = match_ast! {
            match anc {
                ast::Fn(fn_) => hl(sema, fn_.async_token(), fn_.body().map(ast::Expr::BlockExpr)),
                ast::BlockExpr(block_expr) => {
                    let Some(async_token) = block_expr.async_token() else {
                        continue;
                    };

                    // Async blocks act similar to closures. So we want to
                    // highlight their exit points too, but only if we are on
                    // the async token.
                    if async_token == token {
                        let exit_points = hl_exit_points(
                            sema,
                            Some(async_token.clone()),
                            block_expr.clone().into(),
                        );
                        merge_map(&mut res, exit_points);
                    }

                    hl(sema, Some(async_token), Some(block_expr.into()))
                },
                ast::ClosureExpr(closure) => hl(sema, closure.async_token(), closure.body()),
                _ => continue,
            }
        };
        merge_map(&mut res, new_map);
    }

    res.into_iter().map(|(file_id, ranges)| (file_id, ranges.into_iter().collect())).collect()
}

fn cover_range(r0: Option<TextRange>, r1: Option<TextRange>) -> Option<TextRange> {
    match (r0, r1) {
        (Some(r0), Some(r1)) => Some(r0.cover(r1)),
        (Some(range), None) => Some(range),
        (None, Some(range)) => Some(range),
        (None, None) => None,
    }
}

fn find_defs(sema: &Semantics<'_, RootDatabase>, token: SyntaxToken) -> FxHashSet<Definition> {
    sema.descend_into_macros_exact(token)
        .into_iter()
        .filter_map(|token| IdentClass::classify_token(sema, &token))
        .flat_map(IdentClass::definitions_no_ops)
        .collect()
}

fn original_frange(
    db: &dyn db::ExpandDatabase,
    file_id: HirFileId,
    text_range: Option<TextRange>,
) -> Option<FileRange> {
    InFile::new(file_id, text_range?).original_node_file_range_opt(db).map(|(frange, _)| frange)
}

fn merge_map(res: &mut HighlightMap, new: Option<HighlightMap>) {
    let Some(new) = new else {
        return;
    };
    new.into_iter().for_each(|(file_id, ranges)| {
        res.entry(file_id).or_default().extend(ranges);
    });
}

/// Preorder walk all the expression's child expressions.
/// For macro calls, the callback will be called on the expanded expressions after
/// visiting the macro call itself.
struct WalkExpandedExprCtx<'a> {
    sema: &'a Semantics<'a, RootDatabase>,
    depth: usize,
    check_ctx: &'static dyn Fn(&ast::Expr) -> bool,
}

impl<'a> WalkExpandedExprCtx<'a> {
    fn new(sema: &'a Semantics<'a, RootDatabase>) -> Self {
        Self { sema, depth: 0, check_ctx: &is_closure_or_blk_with_modif }
    }

    fn with_check_ctx(&self, check_ctx: &'static dyn Fn(&ast::Expr) -> bool) -> Self {
        Self { check_ctx, ..*self }
    }

    fn walk(&mut self, expr: &ast::Expr, cb: &mut dyn FnMut(usize, ast::Expr)) {
        preorder_expr_with_ctx_checker(expr, self.check_ctx, &mut |ev: WalkEvent<ast::Expr>| {
            match ev {
                syntax::WalkEvent::Enter(expr) => {
                    cb(self.depth, expr.clone());

                    if Self::should_change_depth(&expr) {
                        self.depth += 1;
                    }

                    if let ast::Expr::MacroExpr(expr) = expr
                        && let Some(expanded) =
                            expr.macro_call().and_then(|call| self.sema.expand_macro_call(&call))
                    {
                        match_ast! {
                            match (expanded.value) {
                                ast::MacroStmts(it) => {
                                    self.handle_expanded(it, cb);
                                },
                                ast::Expr(it) => {
                                    self.walk(&it, cb);
                                },
                                _ => {}
                            }
                        }
                    }
                }
                syntax::WalkEvent::Leave(expr) if Self::should_change_depth(&expr) => {
                    self.depth -= 1;
                }
                _ => {}
            }
            false
        })
    }

    fn handle_expanded(&mut self, expanded: ast::MacroStmts, cb: &mut dyn FnMut(usize, ast::Expr)) {
        if let Some(expr) = expanded.expr() {
            self.walk(&expr, cb);
        }

        for stmt in expanded.statements() {
            if let ast::Stmt::ExprStmt(stmt) = stmt
                && let Some(expr) = stmt.expr()
            {
                self.walk(&expr, cb);
            }
        }
    }

    fn should_change_depth(expr: &ast::Expr) -> bool {
        match expr {
            ast::Expr::LoopExpr(_) | ast::Expr::WhileExpr(_) | ast::Expr::ForExpr(_) => true,
            ast::Expr::BlockExpr(blk) if blk.label().is_some() => true,
            _ => false,
        }
    }

    fn is_async_const_block_or_closure(expr: &ast::Expr) -> bool {
        match expr {
            ast::Expr::BlockExpr(b) => matches!(
                b.modifier(),
                Some(ast::BlockModifier::Async(_) | ast::BlockModifier::Const(_))
            ),
            ast::Expr::ClosureExpr(_) => true,
            _ => false,
        }
    }
}

pub(crate) fn highlight_unsafe_points(
    sema: &Semantics<'_, RootDatabase>,
    token: SyntaxToken,
) -> FxHashMap<EditionedFileId, Vec<HighlightedRange>> {
    fn hl(
        sema: &Semantics<'_, RootDatabase>,
        unsafe_token: &SyntaxToken,
        block_expr: Option<ast::BlockExpr>,
    ) -> Option<FxHashMap<EditionedFileId, Vec<HighlightedRange>>> {
        let mut highlights: FxHashMap<EditionedFileId, Vec<_>> = FxHashMap::default();

        let mut push_to_highlights = |file_id, range| {
            if let Some(FileRange { file_id, range }) = original_frange(sema.db, file_id, range) {
                let hrange = HighlightedRange { category: ReferenceCategory::empty(), range };
                highlights.entry(file_id).or_default().push(hrange);
            }
        };

        // highlight unsafe keyword itself
        let unsafe_token_file_id = sema.hir_file_for(&unsafe_token.parent()?);
        push_to_highlights(unsafe_token_file_id, Some(unsafe_token.text_range()));

        // highlight unsafe operations
        if let Some(block) = block_expr
            && let Some(body) = sema.body_for(InFile::new(unsafe_token_file_id, block.syntax()))
        {
            let unsafe_ops = sema.get_unsafe_ops(body);
            for unsafe_op in unsafe_ops {
                push_to_highlights(unsafe_op.file_id, Some(unsafe_op.value.text_range()));
            }
        }

        Some(highlights)
    }

    hl(sema, &token, token.parent().and_then(ast::BlockExpr::cast)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use crate::fixture;

    use super::*;

    const ENABLED_CONFIG: HighlightRelatedConfig = HighlightRelatedConfig {
        break_points: true,
        exit_points: true,
        references: true,
        closure_captures: true,
        yield_points: true,
        branch_exit_points: true,
    };

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
        check_with_config(ra_fixture, ENABLED_CONFIG);
    }

    #[track_caller]
    fn check_with_config(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        config: HighlightRelatedConfig,
    ) {
        let (analysis, pos, annotations) = fixture::annotations(ra_fixture);

        let hls = analysis.highlight_related(config, pos).unwrap().unwrap_or_default();

        let mut expected =
            annotations.into_iter().map(|(r, access)| (r.range, access)).collect::<Vec<_>>();

        let mut actual: Vec<(TextRange, String)> = hls
            .into_iter()
            .map(|hl| {
                (
                    hl.range,
                    hl.category.iter_names().map(|(name, _flag)| name.to_lowercase()).join(","),
                )
            })
            .collect();
        actual.sort_by_key(|(range, _)| range.start());
        expected.sort_by_key(|(range, _)| range.start());

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_hl_unsafe_block() {
        check(
            r#"
fn foo() {
    unsafe fn this_is_unsafe_function() {}

    unsa$0fe {
  //^^^^^^
        let raw_ptr = &42 as *const i32;
        let val = *raw_ptr;
                //^^^^^^^^

        let mut_ptr = &mut 5 as *mut i32;
        *mut_ptr = 10;
      //^^^^^^^^

        this_is_unsafe_function();
      //^^^^^^^^^^^^^^^^^^^^^^^^^
    }

}
"#,
        );
    }

    #[test]
    fn test_hl_tuple_fields() {
        check(
            r#"
struct Tuple(u32, u32);

fn foo(t: Tuple) {
    t.0$0;
   // ^ read
    t.0;
   // ^ read
}
"#,
        );
    }

    #[test]
    fn test_hl_module() {
        check(
            r#"
//- /lib.rs
mod foo$0;
 // ^^^
//- /foo.rs
struct Foo;
"#,
        );
    }

    #[test]
    fn test_hl_self_in_crate_root() {
        check(
            r#"
use crate$0;
  //^^^^^ import
use self;
  //^^^^ import
mod __ {
    use super;
      //^^^^^ import
}
"#,
        );
        check(
            r#"
//- /main.rs crate:main deps:lib
use lib$0;
  //^^^ import
//- /lib.rs crate:lib
"#,
        );
    }

    #[test]
    fn test_hl_self_in_module() {
        check(
            r#"
//- /lib.rs
mod foo;
//- /foo.rs
use self$0;
 // ^^^^ import
"#,
        );
    }

    #[test]
    fn test_hl_local() {
        check(
            r#"
fn foo() {
    let mut bar = 3;
         // ^^^ write
    bar$0;
 // ^^^ read
}
"#,
        );
    }

    #[test]
    fn test_hl_local_in_attr() {
        check(
            r#"
//- proc_macros: identity
#[proc_macros::identity]
fn foo() {
    let mut bar = 3;
         // ^^^ write
    bar$0;
 // ^^^ read
}
"#,
        );
    }

    #[test]
    fn test_multi_macro_usage() {
        check(
            r#"
macro_rules! foo {
    ($ident:ident) => {
        fn $ident() -> $ident { loop {} }
        struct $ident;
    }
}

foo!(bar$0);
  // ^^^
fn foo() {
    let bar: bar = bar();
          // ^^^
                // ^^^
}
"#,
        );
        check(
            r#"
macro_rules! foo {
    ($ident:ident) => {
        fn $ident() -> $ident { loop {} }
        struct $ident;
    }
}

foo!(bar);
  // ^^^
fn foo() {
    let bar: bar$0 = bar();
          // ^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_yield_points() {
        check(
            r#"
pub async fn foo() {
 // ^^^^^
    let x = foo()
        .await$0
      // ^^^^^
        .await;
      // ^^^^^
    || { 0.await };
    (async { 0.await }).await
                     // ^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_yield_points2() {
        check(
            r#"
pub async$0 fn foo() {
 // ^^^^^
    let x = foo()
        .await
      // ^^^^^
        .await;
      // ^^^^^
    || { 0.await };
    (async { 0.await }).await
                     // ^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_exit_points_of_async_blocks() {
        check(
            r#"
pub fn foo() {
    let x = async$0 {
         // ^^^^^
        0.await;
       // ^^^^^
       0?;
     // ^
       return 0;
    // ^^^^^^
       0
    // ^
    };
}
"#,
        );
    }

    #[test]
    fn test_hl_let_else_yield_points() {
        check(
            r#"
pub async fn foo() {
 // ^^^^^
    let x = foo()
        .await$0
      // ^^^^^
        .await;
      // ^^^^^
    || { 0.await };
    let Some(_) = None else {
        foo().await
           // ^^^^^
    };
    (async { 0.await }).await
                     // ^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_yield_nested_fn() {
        check(
            r#"
async fn foo() {
    async fn foo2() {
 // ^^^^^
        async fn foo3() {
            0.await
        }
        0.await$0
       // ^^^^^
    }
    0.await
}
"#,
        );
    }

    #[test]
    fn test_hl_yield_nested_async_blocks() {
        check(
            r#"
async fn foo() {
    (async {
  // ^^^^^
        (async { 0.await }).await$0
                         // ^^^^^
    }).await;
}
"#,
        );
    }

    #[test]
    fn test_hl_exit_points() {
        check(
            r#"
  fn foo() -> u32 {
//^^
    if true {
        return$0 0;
     // ^^^^^^
    }

    0?;
  // ^
    0xDEAD_BEEF
 // ^^^^^^^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_exit_points2() {
        check(
            r#"
  fn foo() ->$0 u32 {
//^^
    if true {
        return 0;
     // ^^^^^^
    }

    0?;
  // ^
    0xDEAD_BEEF
 // ^^^^^^^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_exit_points3() {
        check(
            r#"
  fn$0 foo() -> u32 {
//^^
    if true {
        return 0;
     // ^^^^^^
    }

    0?;
  // ^
    0xDEAD_BEEF
 // ^^^^^^^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_let_else_exit_points() {
        check(
            r#"
  fn$0 foo() -> u32 {
//^^
    let Some(bar) = None else {
        return 0;
     // ^^^^^^
    };

    0?;
  // ^
    0xDEAD_BEEF
 // ^^^^^^^^^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_prefer_ref_over_tail_exit() {
        check(
            r#"
fn foo() -> u32 {
// ^^^
    if true {
        return 0;
    }

    0?;

    foo$0()
 // ^^^
}
"#,
        );
    }

    #[test]
    fn test_hl_never_call_is_exit_point() {
        check(
            r#"
struct Never;
impl Never {
    fn never(self) -> ! { loop {} }
}
macro_rules! never {
    () => { never() }
         // ^^^^^^^
}
fn never() -> ! { loop {} }
  fn foo() ->$0 u32 {
//^^
    never();
 // ^^^^^^^
    never!();
 // ^^^^^^^^

    Never.never();
 // ^^^^^^^^^^^^^

    0
 // ^
}
"#,
        );
    }

    #[test]
    fn test_hl_inner_tail_exit_points() {
        check(
            r#"
  fn foo() ->$0 u32 {
//^^
    if true {
        unsafe {
            return 5;
         // ^^^^^^
            5
         // ^
        }
    } else if false {
        0
     // ^
    } else {
        match 5 {
            6 => 100,
              // ^^^
            7 => loop {
                break 5;
             // ^^^^^
            }
            8 => 'a: loop {
                'b: loop {
                    break 'a 5;
                 // ^^^^^
                    break 'b 5;
                    break 5;
                };
            }
            //
            _ => 500,
              // ^^^
        }
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_inner_tail_exit_points_labeled_block() {
        check(
            r#"
  fn foo() ->$0 u32 {
//^^
    'foo: {
        break 'foo 0;
     // ^^^^^
        loop {
            break;
            break 'foo 0;
         // ^^^^^
        }
        0
     // ^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_inner_tail_exit_points_loops() {
        check(
            r#"
  fn foo() ->$0 u32 {
//^^
    'foo: while { return 0; true } {
               // ^^^^^^
        break 'foo 0;
     // ^^^^^
        return 0;
     // ^^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_loop() {
        check(
            r#"
fn foo() {
    'outer: loop {
 // ^^^^^^^^^^^^
         break;
      // ^^^^^
         'inner: loop {
            break;
            'innermost: loop {
                break 'outer;
             // ^^^^^^^^^^^^
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            break;
        }
        break;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_loop2() {
        check(
            r#"
fn foo() {
    'outer: loop {
        break;
        'inner: loop {
     // ^^^^^^^^^^^^
            break;
         // ^^^^^
            'innermost: loop {
                break 'outer;
                break 'inner;
             // ^^^^^^^^^^^^
            }
            break 'outer;
            break$0;
         // ^^^^^
        }
        break;
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_for() {
        check(
            r#"
fn foo() {
    'outer: for _ in () {
 // ^^^^^^^^^^^
         break;
      // ^^^^^
         'inner: for _ in () {
            break;
            'innermost: for _ in () {
                break 'outer;
             // ^^^^^^^^^^^^
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            break;
        }
        break;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_for_but_not_continue() {
        check(
            r#"
fn foo() {
    'outer: for _ in () {
 // ^^^^^^^^^^^
        break;
     // ^^^^^
        continue;
        'inner: for _ in () {
            break;
            continue;
            'innermost: for _ in () {
                continue 'outer;
                break 'outer;
             // ^^^^^^^^^^^^
                continue 'inner;
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            continue 'outer;
            break;
            continue;
        }
        break;
     // ^^^^^
        continue;
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_continue_for_but_not_break() {
        check(
            r#"
fn foo() {
    'outer: for _ in () {
 // ^^^^^^^^^^^
        break;
        continue;
     // ^^^^^^^^
        'inner: for _ in () {
            break;
            continue;
            'innermost: for _ in () {
                continue 'outer;
             // ^^^^^^^^^^^^^^^
                break 'outer;
                continue 'inner;
                break 'inner;
            }
            break 'outer;
            continue$0 'outer;
         // ^^^^^^^^^^^^^^^
            break;
            continue;
        }
        break;
        continue;
     // ^^^^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_and_continue() {
        check(
            r#"
fn foo() {
    'outer: fo$0r _ in () {
 // ^^^^^^^^^^^
        break;
     // ^^^^^
        continue;
     // ^^^^^^^^
        'inner: for _ in () {
            break;
            continue;
            'innermost: for _ in () {
                continue 'outer;
             // ^^^^^^^^^^^^^^^
                break 'outer;
             // ^^^^^^^^^^^^
                continue 'inner;
                break 'inner;
            }
            break 'outer;
         // ^^^^^^^^^^^^
            continue 'outer;
         // ^^^^^^^^^^^^^^^
            break;
            continue;
        }
        break;
     // ^^^^^
        continue;
     // ^^^^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_while() {
        check(
            r#"
fn foo() {
    'outer: while true {
 // ^^^^^^^^^^^^^
         break;
      // ^^^^^
         'inner: while true {
            break;
            'innermost: while true {
                break 'outer;
             // ^^^^^^^^^^^^
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            break;
        }
        break;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_labeled_block() {
        check(
            r#"
fn foo() {
    'outer: {
 // ^^^^^^^
         break;
      // ^^^^^
         'inner: {
            break;
            'innermost: {
                break 'outer;
             // ^^^^^^^^^^^^
                break 'inner;
            }
            break$0 'outer;
         // ^^^^^^^^^^^^
            break;
        }
        break;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_unlabeled_loop() {
        check(
            r#"
fn foo() {
    loop {
 // ^^^^
        break$0;
     // ^^^^^
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_break_unlabeled_block_in_loop() {
        check(
            r#"
fn foo() {
    loop {
 // ^^^^
        {
            break$0;
         // ^^^^^
        }
    }
}
"#,
        );
    }

    #[test]
    fn test_hl_field_shorthand() {
        check(
            r#"
struct Struct { field: u32 }
              //^^^^^
fn function(field: u32) {
          //^^^^^
    Struct { field$0 }
           //^^^^^ read
}
"#,
        );
    }

    #[test]
    fn test_hl_disabled_ref_local() {
        let config = HighlightRelatedConfig { references: false, ..ENABLED_CONFIG };

        check_with_config(
            r#"
fn foo() {
    let x$0 = 5;
    let y = x * 2;
}
"#,
            config,
        );
    }

    #[test]
    fn test_hl_disabled_ref_local_preserved_break() {
        let config = HighlightRelatedConfig { references: false, ..ENABLED_CONFIG };

        check_with_config(
            r#"
fn foo() {
    let x$0 = 5;
    let y = x * 2;

    loop {
        break;
    }
}
"#,
            config.clone(),
        );

        check_with_config(
            r#"
fn foo() {
    let x = 5;
    let y = x * 2;

    loop$0 {
//  ^^^^
        break;
//      ^^^^^
    }
}
"#,
            config,
        );
    }

    #[test]
    fn test_hl_disabled_ref_local_preserved_yield() {
        let config = HighlightRelatedConfig { references: false, ..ENABLED_CONFIG };

        check_with_config(
            r#"
async fn foo() {
    let x$0 = 5;
    let y = x * 2;

    0.await;
}
"#,
            config.clone(),
        );

        check_with_config(
            r#"
    async fn foo() {
//  ^^^^^
        let x = 5;
        let y = x * 2;

        0.await$0;
//        ^^^^^
}
"#,
            config,
        );
    }

    #[test]
    fn test_hl_disabled_ref_local_preserved_exit() {
        let config = HighlightRelatedConfig { references: false, ..ENABLED_CONFIG };

        check_with_config(
            r#"
fn foo() -> i32 {
    let x$0 = 5;
    let y = x * 2;

    if true {
        return y;
    }

    0?
}
"#,
            config.clone(),
        );

        check_with_config(
            r#"
  fn foo() ->$0 i32 {
//^^
    let x = 5;
    let y = x * 2;

    if true {
        return y;
//      ^^^^^^
    }

    0?
//   ^
"#,
            config,
        );
    }

    #[test]
    fn test_hl_disabled_break() {
        let config = HighlightRelatedConfig { break_points: false, ..ENABLED_CONFIG };

        check_with_config(
            r#"
fn foo() {
    loop {
        break$0;
    }
}
"#,
            config,
        );
    }

    #[test]
    fn test_hl_disabled_yield() {
        let config = HighlightRelatedConfig { yield_points: false, ..ENABLED_CONFIG };

        check_with_config(
            r#"
async$0 fn foo() {
    0.await;
}
"#,
            config,
        );
    }

    #[test]
    fn test_hl_disabled_exit() {
        let config = HighlightRelatedConfig { exit_points: false, ..ENABLED_CONFIG };

        check_with_config(
            r#"
fn foo() ->$0 i32 {
    if true {
        return -1;
    }

    42
}"#,
            config,
        );
    }

    #[test]
    fn test_hl_multi_local() {
        check(
            r#"
fn foo((
    foo$0
  //^^^
    | foo
    //^^^
    | foo
    //^^^
): ()) {
    foo;
  //^^^read
    let foo;
}
"#,
        );
        check(
            r#"
fn foo((
    foo
  //^^^
    | foo$0
    //^^^
    | foo
    //^^^
): ()) {
    foo;
  //^^^read
    let foo;
}
"#,
        );
        check(
            r#"
fn foo((
    foo
  //^^^
    | foo
    //^^^
    | foo
    //^^^
): ()) {
    foo$0;
  //^^^read
    let foo;
}
"#,
        );
    }

    #[test]
    fn test_hl_trait_impl_methods() {
        check(
            r#"
trait Trait {
    fn func$0(self) {}
     //^^^^
}

impl Trait for () {
    fn func(self) {}
     //^^^^
}

fn main() {
    <()>::func(());
        //^^^^
    ().func();
     //^^^^
}
"#,
        );
        check(
            r#"
trait Trait {
    fn func(self) {}
}

impl Trait for () {
    fn func$0(self) {}
     //^^^^
}

fn main() {
    <()>::func(());
        //^^^^
    ().func();
     //^^^^
}
"#,
        );
        check(
            r#"
trait Trait {
    fn func(self) {}
}

impl Trait for () {
    fn func(self) {}
     //^^^^
}

fn main() {
    <()>::func(());
        //^^^^
    ().func$0();
     //^^^^
}
"#,
        );
    }

    #[test]
    fn test_assoc_type_highlighting() {
        check(
            r#"
trait Trait {
    type Output;
      // ^^^^^^
}
impl Trait for () {
    type Output$0 = ();
      // ^^^^^^
}
"#,
        );
    }

    #[test]
    fn test_closure_capture_pipe() {
        check(
            r#"
fn f() {
    let x = 1;
    //  ^
    let c = $0|y| x + y;
    //          ^ read
}
"#,
        );
    }

    #[test]
    fn test_closure_capture_move() {
        check(
            r#"
fn f() {
    let x = 1;
    //  ^
    let c = move$0 |y| x + y;
    //               ^ read
}
"#,
        );
    }

    #[test]
    fn test_trait_highlights_assoc_item_uses() {
        check(
            r#"
trait Super {
    type SuperT;
}
trait Foo: Super {
    //^^^
    type T;
    const C: usize;
    fn f() {}
    fn m(&self) {}
}
impl Foo for i32 {
   //^^^
    type T = i32;
    const C: usize = 0;
    fn f() {}
    fn m(&self) {}
}
fn f<T: Foo$0>(t: T) {
      //^^^
    let _: T::SuperT;
            //^^^^^^
    let _: T::T;
            //^
    t.m();
    //^
    T::C;
     //^
    T::f();
     //^
}

fn f2<T: Foo>(t: T) {
       //^^^
    let _: T::T;
    t.m();
    T::C;
    T::f();
}
"#,
        );
    }

    #[test]
    fn test_trait_highlights_assoc_item_uses_use_tree() {
        check(
            r#"
use Foo$0;
 // ^^^ import
trait Super {
    type SuperT;
}
trait Foo: Super {
    //^^^
    type T;
    const C: usize;
    fn f() {}
    fn m(&self) {}
}
impl Foo for i32 {
   //^^^
    type T = i32;
      // ^
    const C: usize = 0;
       // ^
    fn f() {}
    // ^
    fn m(&self) {}
    // ^
}
fn f<T: Foo>(t: T) {
      //^^^
    let _: T::SuperT;
    let _: T::T;
            //^
    t.m();
    //^
    T::C;
     //^
    T::f();
     //^
}
"#,
        );
    }

    #[test]
    fn implicit_format_args() {
        check(
            r#"
//- minicore: fmt
fn test() {
    let a = "foo";
     // ^
    format_args!("hello {a} {a$0} {}", a);
                      // ^read
                          // ^read
                                  // ^read
}
"#,
        );
    }

    #[test]
    fn return_in_macros() {
        check(
            r#"
macro_rules! N {
    ($i:ident, $x:expr, $blk:expr) => {
        for $i in 0..$x {
            $blk
        }
    };
}

fn main() {
    fn f() {
 // ^^
        N!(i, 5, {
            println!("{}", i);
            return$0;
         // ^^^^^^
        });

        for i in 1..5 {
            return;
         // ^^^^^^
        }
       (|| {
            return;
        })();
    }
}
"#,
        )
    }

    #[test]
    fn return_in_closure() {
        check(
            r#"
macro_rules! N {
    ($i:ident, $x:expr, $blk:expr) => {
        for $i in 0..$x {
            $blk
        }
    };
}

fn main() {
    fn f() {
        N!(i, 5, {
            println!("{}", i);
            return;
        });

        for i in 1..5 {
            return;
        }
       (|| {
     // ^
            return$0;
         // ^^^^^^
        })();
    }
}
"#,
        )
    }

    #[test]
    fn return_in_try() {
        check(
            r#"
fn main() {
    fn f() {
 // ^^
        try {
            return$0;
         // ^^^^^^
        }

        return;
     // ^^^^^^
    }
}
"#,
        )
    }

    #[test]
    fn break_in_try() {
        check(
            r#"
fn main() {
    for i in 1..100 {
 // ^^^
        let x: Result<(), ()> = try {
            break$0;
         // ^^^^^
        };
    }
}
"#,
        )
    }

    #[test]
    fn no_highlight_on_return_in_macro_call() {
        check(
            r#"
//- minicore:include
//- /lib.rs
macro_rules! M {
    ($blk:expr) => {
        $blk
    };
}

fn main() {
    fn f() {
 // ^^
        M!({ return$0; });
          // ^^^^^^
     // ^^^^^^^^^^^^^^^

        include!("a.rs")
     // ^^^^^^^^^^^^^^^^
    }
}

//- /a.rs
{
    return;
}
"#,
        )
    }

    #[test]
    fn nested_match() {
        check(
            r#"
fn main() {
    match$0 0 {
 // ^^^^^
        0 => match 1 {
            1 => 2,
              // ^
            _ => 3,
              // ^
        },
        _ => 4,
          // ^
    }
}
"#,
        )
    }

    #[test]
    fn single_arm_highlight() {
        check(
            r#"
fn main() {
    match 0 {
        0 =>$0 {
       // ^^
            let x = 1;
            x
         // ^
        }
        _ => 2,
    }
}
"#,
        )
    }

    #[test]
    fn no_branches_when_disabled() {
        let config = HighlightRelatedConfig { branch_exit_points: false, ..ENABLED_CONFIG };
        check_with_config(
            r#"
fn main() {
    match$0 0 {
        0 => 1,
        _ => 2,
    }
}
"#,
            config,
        );
    }

    #[test]
    fn asm() {
        check(
            r#"
//- minicore: asm
#[inline]
pub unsafe fn bootstrap() -> ! {
    builtin#asm(
        "blabla",
        "mrs {tmp}, CONTROL",
           // ^^^ read
        "blabla",
        "bics {tmp}, {spsel}",
            // ^^^ read
        "blabla",
        "msr CONTROL, {tmp}",
                    // ^^^ read
        "blabla",
        tmp$0 = inout(reg) 0,
     // ^^^
        aaa = in(reg) 2,
        aaa = in(reg) msp,
        aaa = in(reg) rv,
        options(noreturn, nomem, nostack),
    );
}
"#,
        )
    }

    #[test]
    fn complex_arms_highlight() {
        check(
            r#"
fn calculate(n: i32) -> i32 { n * 2 }

fn main() {
    match$0 Some(1) {
 // ^^^^^
        Some(x) => match x {
            0 => { let y = x; y },
                           // ^
            1 => calculate(x),
               //^^^^^^^^^^^^
            _ => (|| 6)(),
              // ^^^^^^^^
        },
        None => loop {
            break 5;
         // ^^^^^^^
        },
    }
}
"#,
        )
    }

    #[test]
    fn match_in_macro_highlight() {
        check(
            r#"
macro_rules! M {
    ($e:expr) => { $e };
}

fn main() {
    M!{
        match$0 Some(1) {
     // ^^^^^
            Some(x) => x,
                    // ^
            None => 0,
                 // ^
        }
    }
}
"#,
        )
    }

    #[test]
    fn match_in_macro_highlight_2() {
        check(
            r#"
macro_rules! match_ast {
    (match $node:ident { $($tt:tt)* }) => { $crate::match_ast!(match ($node) { $($tt)* }) };

    (match ($node:expr) {
        $( $( $path:ident )::+ ($it:pat) => $res:expr, )*
        _ => $catch_all:expr $(,)?
    }) => {{
        $( if let Some($it) = $($path::)+cast($node.clone()) { $res } else )*
        { $catch_all }
    }};
}

fn main() {
    match_ast! {
        match$0 Some(1) {
            Some(x) => x,
        }
    }
}
            "#,
        );
    }

    #[test]
    fn nested_if_else() {
        check(
            r#"
fn main() {
    if$0 true {
 // ^^
        if false {
            1
         // ^
        } else {
            2
         // ^
        }
    } else {
        3
     // ^
    }
}
"#,
        )
    }

    #[test]
    fn if_else_if_highlight() {
        check(
            r#"
fn main() {
    if$0 true {
 // ^^
        1
     // ^
    } else if false {
        // ^^
        2
     // ^
    } else {
        3
     // ^
    }
}
"#,
        )
    }

    #[test]
    fn complex_if_branches() {
        check(
            r#"
fn calculate(n: i32) -> i32 { n * 2 }

fn main() {
    if$0 true {
 // ^^
        let x = 5;
        calculate(x)
     // ^^^^^^^^^^^^
    } else if false {
        // ^^
        (|| 10)()
     // ^^^^^^^^^
    } else {
        loop {
            break 15;
         // ^^^^^^^^
        }
    }
}
"#,
        )
    }

    #[test]
    fn if_in_macro_highlight() {
        check(
            r#"
macro_rules! M {
    ($e:expr) => { $e };
}

fn main() {
    M!{
        if$0 true {
     // ^^
            5
         // ^
        } else {
            10
         // ^^
        }
    }
}
"#,
        )
    }

    #[test]
    fn match_in_macro() {
        // We should not highlight the outer `match` expression.
        check(
            r#"
macro_rules! M {
    (match) => { 1 };
}

fn main() {
    match Some(1) {
        Some(x) => x,
        None => {
            M!(match$0)
        }
    }
}
            "#,
        )
    }

    #[test]
    fn labeled_block_tail_expr() {
        check(
            r#"
fn foo() {
    'a: {
 // ^^^
        if true { break$0 'a 0; }
               // ^^^^^^^^
        5
     // ^
    }
}
"#,
        );
    }

    #[test]
    fn labeled_block_tail_expr_2() {
        check(
            r#"
fn foo() {
    let _ = 'b$0lk: {
         // ^^^^
        let x = 1;
        if true { break 'blk 42; }
                     // ^^^^
        if false { break 'blk 24; }
                      // ^^^^
        100
     // ^^^
    };
}
"#,
        );
    }
}
