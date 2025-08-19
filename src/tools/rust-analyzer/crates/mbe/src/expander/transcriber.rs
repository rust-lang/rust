//! Transcriber takes a template, like `fn $ident() {}`, a set of bindings like
//! `$ident => foo`, interpolates variables in the template, to get `fn foo() {}`

use intern::{Symbol, sym};
use span::{Edition, Span};
use tt::{Delimiter, TopSubtreeBuilder, iter::TtElement};

use crate::{
    ExpandError, ExpandErrorKind, ExpandResult, MetaTemplate,
    expander::{Binding, Bindings, Fragment},
    parser::{ConcatMetaVarExprElem, MetaVarKind, Op, RepeatKind, Separator},
};

impl<'t> Bindings<'t> {
    fn get(&self, name: &Symbol, span: Span) -> Result<&Binding<'t>, ExpandError> {
        match self.inner.get(name) {
            Some(binding) => Ok(binding),
            None => Err(ExpandError::new(
                span,
                ExpandErrorKind::UnresolvedBinding(Box::new(Box::from(name.as_str()))),
            )),
        }
    }

    fn get_fragment(
        &self,
        name: &Symbol,
        mut span: Span,
        nesting: &mut [NestingState],
        marker: impl Fn(&mut Span),
    ) -> Result<Fragment<'t>, ExpandError> {
        macro_rules! binding_err {
            ($($arg:tt)*) => { ExpandError::binding_error(span, format!($($arg)*)) };
        }

        let mut b = self.get(name, span)?;
        for nesting_state in nesting.iter_mut() {
            nesting_state.hit = true;
            b = match b {
                Binding::Fragment(_) => break,
                Binding::Missing(_) => {
                    nesting_state.at_end = true;
                    break;
                }
                Binding::Nested(bs) => bs.get(nesting_state.idx).ok_or_else(|| {
                    nesting_state.at_end = true;
                    binding_err!("could not find nested binding `{name}`")
                })?,
                Binding::Empty => {
                    nesting_state.at_end = true;
                    return Err(binding_err!("could not find empty binding `{name}`"));
                }
            };
        }
        match b {
            Binding::Fragment(f) => Ok(f.clone()),
            // emit some reasonable default expansion for missing bindings,
            // this gives better recovery than emitting the `$fragment-name` verbatim
            Binding::Missing(it) => Ok({
                marker(&mut span);
                let mut builder = TopSubtreeBuilder::new(tt::Delimiter::invisible_spanned(span));
                match it {
                    MetaVarKind::Stmt => {
                        builder.push(tt::Leaf::Punct(tt::Punct {
                            span,
                            char: ';',
                            spacing: tt::Spacing::Alone,
                        }));
                    }
                    MetaVarKind::Block => {
                        builder.open(tt::DelimiterKind::Brace, span);
                        builder.close(span);
                    }
                    // FIXME: Meta and Item should get proper defaults
                    MetaVarKind::Meta | MetaVarKind::Item | MetaVarKind::Tt | MetaVarKind::Vis => {}
                    MetaVarKind::Path
                    | MetaVarKind::Ty
                    | MetaVarKind::Pat
                    | MetaVarKind::PatParam
                    | MetaVarKind::Expr(_)
                    | MetaVarKind::Ident => {
                        builder.push(tt::Leaf::Ident(tt::Ident {
                            sym: sym::missing,
                            span,
                            is_raw: tt::IdentIsRaw::No,
                        }));
                    }
                    MetaVarKind::Lifetime => {
                        builder.extend([
                            tt::Leaf::Punct(tt::Punct {
                                char: '\'',
                                span,
                                spacing: tt::Spacing::Joint,
                            }),
                            tt::Leaf::Ident(tt::Ident {
                                sym: sym::missing,
                                span,
                                is_raw: tt::IdentIsRaw::No,
                            }),
                        ]);
                    }
                    MetaVarKind::Literal => {
                        builder.push(tt::Leaf::Ident(tt::Ident {
                            sym: sym::missing,
                            span,
                            is_raw: tt::IdentIsRaw::No,
                        }));
                    }
                }
                Fragment::TokensOwned(builder.build())
            }),
            Binding::Nested(_) => {
                Err(binding_err!("expected simple binding, found nested binding `{name}`"))
            }
            Binding::Empty => {
                Err(binding_err!("expected simple binding, found empty binding `{name}`"))
            }
        }
    }
}

pub(super) fn transcribe(
    template: &MetaTemplate,
    bindings: &Bindings<'_>,
    marker: impl Fn(&mut Span) + Copy,
    call_site: Span,
) -> ExpandResult<tt::TopSubtree<Span>> {
    let mut ctx = ExpandCtx { bindings, nesting: Vec::new(), call_site };
    let mut builder = tt::TopSubtreeBuilder::new(tt::Delimiter::invisible_spanned(ctx.call_site));
    expand_subtree(&mut ctx, template, &mut builder, marker).map(|()| builder.build())
}

#[derive(Debug)]
struct NestingState {
    idx: usize,
    /// `hit` is currently necessary to tell `expand_repeat` if it should stop
    /// because there is no variable in use by the current repetition
    hit: bool,
    /// `at_end` is currently necessary to tell `expand_repeat` if it should stop
    /// because there is no more value available for the current repetition
    at_end: bool,
}

#[derive(Debug)]
struct ExpandCtx<'a> {
    bindings: &'a Bindings<'a>,
    nesting: Vec<NestingState>,
    call_site: Span,
}

fn expand_subtree_with_delimiter(
    ctx: &mut ExpandCtx<'_>,
    template: &MetaTemplate,
    builder: &mut tt::TopSubtreeBuilder<Span>,
    delimiter: Option<Delimiter<Span>>,
    marker: impl Fn(&mut Span) + Copy,
) -> ExpandResult<()> {
    let delimiter = delimiter.unwrap_or_else(|| tt::Delimiter::invisible_spanned(ctx.call_site));
    builder.open(delimiter.kind, delimiter.open);
    let result = expand_subtree(ctx, template, builder, marker);
    builder.close(delimiter.close);
    result
}

fn expand_subtree(
    ctx: &mut ExpandCtx<'_>,
    template: &MetaTemplate,
    builder: &mut tt::TopSubtreeBuilder<Span>,
    marker: impl Fn(&mut Span) + Copy,
) -> ExpandResult<()> {
    let mut err = None;
    'ops: for op in template.iter() {
        match op {
            Op::Literal(it) => builder.push(tt::Leaf::from({
                let mut it = it.clone();
                marker(&mut it.span);
                it
            })),
            Op::Ident(it) => builder.push(tt::Leaf::from({
                let mut it = it.clone();
                marker(&mut it.span);
                it
            })),
            Op::Punct(puncts) => {
                builder.extend(puncts.iter().map(|punct| {
                    tt::Leaf::from({
                        let mut it = *punct;
                        marker(&mut it.span);
                        it
                    })
                }));
            }
            Op::Subtree { tokens, delimiter } => {
                let mut delimiter = *delimiter;
                marker(&mut delimiter.open);
                marker(&mut delimiter.close);
                let ExpandResult { value: (), err: e } =
                    expand_subtree_with_delimiter(ctx, tokens, builder, Some(delimiter), marker);
                err = err.or(e);
            }
            Op::Var { name, id, .. } => {
                let ExpandResult { value: (), err: e } =
                    expand_var(ctx, name, *id, builder, marker);
                err = err.or(e);
            }
            Op::Repeat { tokens: subtree, kind, separator } => {
                let ExpandResult { value: (), err: e } =
                    expand_repeat(ctx, subtree, *kind, separator.as_deref(), builder, marker);
                err = err.or(e);
            }
            Op::Ignore { name, id } => {
                // Expand the variable, but ignore the result. This registers the repetition count.
                let e = ctx.bindings.get_fragment(name, *id, &mut ctx.nesting, marker).err();
                // FIXME: The error gets dropped if there were any previous errors.
                // This should be reworked in a way where the errors can be combined
                // and reported rather than storing the first error encountered.
                err = err.or(e);
            }
            Op::Index { depth } => {
                let index =
                    ctx.nesting.get(ctx.nesting.len() - 1 - depth).map_or(0, |nest| nest.idx);
                builder.push(tt::Leaf::Literal(tt::Literal {
                    symbol: Symbol::integer(index),
                    span: ctx.call_site,
                    kind: tt::LitKind::Integer,
                    suffix: None,
                }));
            }
            Op::Len { depth } => {
                let length = ctx.nesting.get(ctx.nesting.len() - 1 - depth).map_or(0, |_nest| {
                    // FIXME: to be implemented
                    0
                });
                builder.push(tt::Leaf::Literal(tt::Literal {
                    symbol: Symbol::integer(length),
                    span: ctx.call_site,
                    kind: tt::LitKind::Integer,
                    suffix: None,
                }));
            }
            Op::Count { name, depth } => {
                let mut binding = match ctx.bindings.get(name, ctx.call_site) {
                    Ok(b) => b,
                    Err(e) => {
                        err = err.or(Some(e));
                        continue;
                    }
                };
                for state in ctx.nesting.iter_mut() {
                    state.hit = true;
                    match binding {
                        Binding::Fragment(_) | Binding::Missing(_) => {
                            // `count()` will report an error.
                            break;
                        }
                        Binding::Nested(bs) => {
                            if let Some(b) = bs.get(state.idx) {
                                binding = b;
                            } else {
                                state.at_end = true;
                                continue 'ops;
                            }
                        }
                        Binding::Empty => {
                            state.at_end = true;
                            // FIXME: Breaking here and proceeding to `count()` isn't the most
                            // correct thing to do here. This could be a binding of some named
                            // fragment which we don't know the depth of, so `count()` will just
                            // return 0 for this no matter what `depth` is. See test
                            // `count_interaction_with_empty_binding` for example.
                            break;
                        }
                    }
                }

                let res = count(binding, 0, depth.unwrap_or(0));

                builder.push(tt::Leaf::Literal(tt::Literal {
                    symbol: Symbol::integer(res),
                    span: ctx.call_site,
                    suffix: None,
                    kind: tt::LitKind::Integer,
                }));
            }
            Op::Concat { elements, span: concat_span } => {
                let mut concatenated = String::new();
                for element in elements {
                    match element {
                        ConcatMetaVarExprElem::Ident(ident) => {
                            concatenated.push_str(ident.sym.as_str())
                        }
                        ConcatMetaVarExprElem::Literal(lit) => {
                            // FIXME: This isn't really correct wrt. escaping, but that's what rustc does and anyway
                            // escaping is used most of the times for characters that are invalid in identifiers.
                            concatenated.push_str(lit.symbol.as_str())
                        }
                        ConcatMetaVarExprElem::Var(var) => {
                            // Handling of repetitions in `${concat}` isn't fleshed out in rustc, so we currently
                            // err at it.
                            // FIXME: Do what rustc does for repetitions.
                            let var_value = match ctx.bindings.get_fragment(
                                &var.sym,
                                var.span,
                                &mut ctx.nesting,
                                marker,
                            ) {
                                Ok(var) => var,
                                Err(e) => {
                                    if err.is_none() {
                                        err = Some(e);
                                    };
                                    continue;
                                }
                            };
                            let values = match &var_value {
                                Fragment::Tokens(tokens) => {
                                    let mut iter = tokens.iter();
                                    (iter.next(), iter.next())
                                }
                                Fragment::TokensOwned(tokens) => {
                                    let mut iter = tokens.iter();
                                    (iter.next(), iter.next())
                                }
                                _ => (None, None),
                            };
                            let value = match values {
                                (Some(TtElement::Leaf(tt::Leaf::Ident(ident))), None) => {
                                    ident.sym.as_str()
                                }
                                (Some(TtElement::Leaf(tt::Leaf::Literal(lit))), None) => {
                                    lit.symbol.as_str()
                                }
                                _ => {
                                    if err.is_none() {
                                        err = Some(ExpandError::binding_error(
                                            var.span,
                                            "metavariables of `${concat(..)}` must be of type `ident`, `literal` or `tt`",
                                        ))
                                    }
                                    continue;
                                }
                            };
                            concatenated.push_str(value);
                        }
                    }
                }

                // `${concat}` span comes from the macro (at least for now).
                // See https://github.com/rust-lang/rust/blob/b0af276da341/compiler/rustc_expand/src/mbe/transcribe.rs#L724-L726.
                let mut result_span = *concat_span;
                marker(&mut result_span);

                // FIXME: NFC normalize the result.
                if !rustc_lexer::is_ident(&concatenated) {
                    if err.is_none() {
                        err = Some(ExpandError::binding_error(
                            *concat_span,
                            "`${concat(..)}` is not generating a valid identifier",
                        ));
                    }
                    // Insert a dummy identifier for better parsing.
                    concatenated.clear();
                    concatenated.push_str("__ra_concat_dummy");
                }

                let needs_raw =
                    parser::SyntaxKind::from_keyword(&concatenated, Edition::LATEST).is_some();
                let is_raw = if needs_raw { tt::IdentIsRaw::Yes } else { tt::IdentIsRaw::No };
                builder.push(tt::Leaf::Ident(tt::Ident {
                    is_raw,
                    span: result_span,
                    sym: Symbol::intern(&concatenated),
                }));
            }
        }
    }
    ExpandResult { value: (), err }
}

fn expand_var(
    ctx: &mut ExpandCtx<'_>,
    v: &Symbol,
    id: Span,
    builder: &mut tt::TopSubtreeBuilder<Span>,
    marker: impl Fn(&mut Span) + Copy,
) -> ExpandResult<()> {
    // We already handle $crate case in mbe parser
    debug_assert!(*v != sym::crate_);

    match ctx.bindings.get_fragment(v, id, &mut ctx.nesting, marker) {
        Ok(fragment) => {
            match fragment {
                // rustc spacing is not like ours. Ours is like proc macros', it dictates how puncts will actually be joined.
                // rustc uses them mostly for pretty printing. So we have to deviate a bit from what rustc does here.
                // Basically, a metavariable can never be joined with whatever after it.
                Fragment::Tokens(tt) => builder.extend_with_tt_alone(tt.strip_invisible()),
                Fragment::TokensOwned(tt) => {
                    builder.extend_with_tt_alone(tt.view().strip_invisible())
                }
                Fragment::Expr(sub) => {
                    let sub = sub.strip_invisible();
                    let mut span = id;
                    marker(&mut span);
                    let wrap_in_parens = !matches!(sub.flat_tokens(), [tt::TokenTree::Leaf(_)])
                        && sub.try_into_subtree().is_none_or(|it| {
                            it.top_subtree().delimiter.kind == tt::DelimiterKind::Invisible
                        });
                    if wrap_in_parens {
                        builder.open(tt::DelimiterKind::Parenthesis, span);
                    }
                    builder.extend_with_tt_alone(sub);
                    if wrap_in_parens {
                        builder.close(span);
                    }
                }
                Fragment::Path(tt) => fix_up_and_push_path_tt(ctx, builder, tt),
                Fragment::Empty => (),
            };
            ExpandResult::ok(())
        }
        Err(e) if matches!(e.inner.1, ExpandErrorKind::UnresolvedBinding(_)) => {
            // Note that it is possible to have a `$var` inside a macro which is not bound.
            // For example:
            // ```
            // macro_rules! foo {
            //     ($a:ident, $b:ident, $c:tt) => {
            //         macro_rules! bar {
            //             ($bi:ident) => {
            //                 fn $bi() -> u8 {$c}
            //             }
            //         }
            //     }
            // ```
            // We just treat it a normal tokens
            builder.extend([
                tt::Leaf::from(tt::Punct { char: '$', spacing: tt::Spacing::Alone, span: id }),
                tt::Leaf::from(tt::Ident { sym: v.clone(), span: id, is_raw: tt::IdentIsRaw::No }),
            ]);
            ExpandResult::ok(())
        }
        Err(e) => ExpandResult::only_err(e),
    }
}

fn expand_repeat(
    ctx: &mut ExpandCtx<'_>,
    template: &MetaTemplate,
    kind: RepeatKind,
    separator: Option<&Separator>,
    builder: &mut tt::TopSubtreeBuilder<Span>,
    marker: impl Fn(&mut Span) + Copy,
) -> ExpandResult<()> {
    ctx.nesting.push(NestingState { idx: 0, at_end: false, hit: false });
    // Dirty hack to make macro-expansion terminate.
    // This should be replaced by a proper macro-by-example implementation
    let limit = 65536;
    let mut counter = 0;
    let mut err = None;

    let initial_restore_point = builder.restore_point();
    let mut restore_point = builder.restore_point();
    loop {
        let ExpandResult { value: (), err: e } =
            expand_subtree_with_delimiter(ctx, template, builder, None, marker);
        let nesting_state = ctx.nesting.last_mut().unwrap();
        if nesting_state.at_end || !nesting_state.hit {
            break;
        }
        nesting_state.idx += 1;
        nesting_state.hit = false;

        builder.remove_last_subtree_if_invisible();

        restore_point = builder.restore_point();

        counter += 1;
        if counter == limit {
            // FIXME: This is a bug here, we get here when we shouldn't, see https://github.com/rust-lang/rust-analyzer/issues/18910.
            // If we don't restore we emit a lot of nodes which causes a stack overflow down the road. For now just ignore them,
            // there is always an error here anyway.
            builder.restore(initial_restore_point);
            err = Some(ExpandError::new(ctx.call_site, ExpandErrorKind::LimitExceeded));
            break;
        }

        if e.is_some() {
            err = err.or(e);
            continue;
        }

        if let Some(sep) = separator {
            match sep {
                Separator::Ident(ident) => builder.push(tt::Leaf::from(ident.clone())),
                Separator::Literal(lit) => builder.push(tt::Leaf::from(lit.clone())),
                Separator::Puncts(puncts) => {
                    for &punct in puncts {
                        builder.push(tt::Leaf::from(punct));
                    }
                }
                Separator::Lifetime(punct, ident) => {
                    builder.push(tt::Leaf::from(*punct));
                    builder.push(tt::Leaf::from(ident.clone()));
                }
            };
        }

        if RepeatKind::ZeroOrOne == kind {
            break;
        }
    }
    // Lose the last separator and last after-the-end round.
    builder.restore(restore_point);

    ctx.nesting.pop().unwrap();

    // Check if it is a single token subtree without any delimiter
    // e.g {Delimiter:None> ['>'] /Delimiter:None>}

    if RepeatKind::OneOrMore == kind && counter == 0 && err.is_none() {
        err = Some(ExpandError::new(ctx.call_site, ExpandErrorKind::UnexpectedToken));
    }
    ExpandResult { value: (), err }
}

/// Inserts the path separator `::` between an identifier and its following generic
/// argument list, and then pushes into the buffer. See [`Fragment::Path`] for why
/// we need this fixup.
fn fix_up_and_push_path_tt(
    ctx: &ExpandCtx<'_>,
    builder: &mut tt::TopSubtreeBuilder<Span>,
    subtree: tt::TokenTreesView<'_, Span>,
) {
    let mut prev_was_ident = false;
    // Note that we only need to fix up the top-level `TokenTree`s because the
    // context of the paths in the descendant `Subtree`s won't be changed by the
    // mbe transcription.
    let mut iter = subtree.iter();
    while let Some(tt) = iter.next_as_view() {
        if prev_was_ident {
            // Pedantically, `(T) -> U` in `FnOnce(T) -> U` is treated as a generic
            // argument list and thus needs `::` between it and `FnOnce`. However in
            // today's Rust this type of path *semantically* cannot appear as a
            // top-level expression-context path, so we can safely ignore it.
            if let [tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: '<', .. }))] =
                tt.flat_tokens()
            {
                builder.extend([
                    tt::Leaf::Punct(tt::Punct {
                        char: ':',
                        spacing: tt::Spacing::Joint,
                        span: ctx.call_site,
                    }),
                    tt::Leaf::Punct(tt::Punct {
                        char: ':',
                        spacing: tt::Spacing::Alone,
                        span: ctx.call_site,
                    }),
                ]);
            }
        }
        prev_was_ident = matches!(tt.flat_tokens(), [tt::TokenTree::Leaf(tt::Leaf::Ident(_))]);
        builder.extend_with_tt(tt);
    }
}

/// Handles `${count(t, depth)}`. `our_depth` is the recursion depth and `count_depth` is the depth
/// defined by the metavar expression.
fn count(binding: &Binding<'_>, depth_curr: usize, depth_max: usize) -> usize {
    match binding {
        Binding::Nested(bs) => {
            if depth_curr == depth_max {
                bs.len()
            } else {
                bs.iter().map(|b| count(b, depth_curr + 1, depth_max)).sum()
            }
        }
        Binding::Empty => 0,
        Binding::Fragment(_) | Binding::Missing(_) => 1,
    }
}
