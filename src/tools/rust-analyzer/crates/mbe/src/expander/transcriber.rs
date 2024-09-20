//! Transcriber takes a template, like `fn $ident() {}`, a set of bindings like
//! `$ident => foo`, interpolates variables in the template, to get `fn foo() {}`

use intern::{sym, Symbol};
use span::{Edition, Span};
use tt::Delimiter;

use crate::{
    expander::{Binding, Bindings, Fragment},
    parser::{ConcatMetaVarExprElem, MetaVarKind, Op, RepeatKind, Separator},
    ExpandError, ExpandErrorKind, ExpandResult, MetaTemplate,
};

impl Bindings {
    fn get(&self, name: &Symbol, span: Span) -> Result<&Binding, ExpandError> {
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
    ) -> Result<Fragment, ExpandError> {
        macro_rules! binding_err {
            ($($arg:tt)*) => { ExpandError::binding_error(span, format!($($arg)*)) };
        }

        let mut b = self.get(name, span)?;
        for nesting_state in nesting.iter_mut() {
            nesting_state.hit = true;
            b = match b {
                Binding::Fragment(_) => break,
                Binding::Missing(_) => break,
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
            Binding::Fragment(f @ (Fragment::Path(sub) | Fragment::Expr(sub))) => {
                let tt::Subtree { delimiter, token_trees } = sub;
                marker(&mut span);
                let subtree = tt::Subtree {
                    delimiter: tt::Delimiter {
                        // FIXME split span
                        open: span,
                        close: span,
                        kind: delimiter.kind,
                    },
                    token_trees: token_trees.clone(),
                };
                Ok(match f {
                    Fragment::Tokens(_) | Fragment::Empty => unreachable!(),
                    Fragment::Expr(_) => Fragment::Expr,
                    Fragment::Path(_) => Fragment::Path,
                }(subtree))
            }
            Binding::Fragment(it @ (Fragment::Tokens(_) | Fragment::Empty)) => Ok(it.clone()),
            // emit some reasonable default expansion for missing bindings,
            // this gives better recovery than emitting the `$fragment-name` verbatim
            Binding::Missing(it) => Ok({
                marker(&mut span);
                match it {
                    MetaVarKind::Stmt => {
                        Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
                            span,
                            char: ';',
                            spacing: tt::Spacing::Alone,
                        })))
                    }
                    MetaVarKind::Block => Fragment::Tokens(tt::TokenTree::Subtree(tt::Subtree {
                        delimiter: tt::Delimiter {
                            open: span,
                            close: span,
                            kind: tt::DelimiterKind::Brace,
                        },
                        token_trees: Box::new([]),
                    })),
                    // FIXME: Meta and Item should get proper defaults
                    MetaVarKind::Meta | MetaVarKind::Item | MetaVarKind::Tt | MetaVarKind::Vis => {
                        Fragment::Empty
                    }
                    MetaVarKind::Path
                    | MetaVarKind::Ty
                    | MetaVarKind::Pat
                    | MetaVarKind::PatParam
                    | MetaVarKind::Expr(_)
                    | MetaVarKind::Ident => {
                        Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                            sym: sym::missing.clone(),
                            span,
                            is_raw: tt::IdentIsRaw::No,
                        })))
                    }
                    MetaVarKind::Lifetime => {
                        Fragment::Tokens(tt::TokenTree::Subtree(tt::Subtree {
                            delimiter: tt::Delimiter::invisible_spanned(span),
                            token_trees: Box::new([
                                tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
                                    char: '\'',
                                    span,
                                    spacing: tt::Spacing::Joint,
                                })),
                                tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                                    sym: sym::missing.clone(),
                                    span,
                                    is_raw: tt::IdentIsRaw::No,
                                })),
                            ]),
                        }))
                    }
                    MetaVarKind::Literal => {
                        Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                            sym: sym::missing.clone(),
                            span,
                            is_raw: tt::IdentIsRaw::No,
                        })))
                    }
                }
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
    bindings: &Bindings,
    marker: impl Fn(&mut Span) + Copy,
    call_site: Span,
) -> ExpandResult<tt::Subtree<Span>> {
    let mut ctx = ExpandCtx { bindings, nesting: Vec::new(), call_site };
    let mut arena: Vec<tt::TokenTree<Span>> = Vec::new();
    expand_subtree(&mut ctx, template, None, &mut arena, marker)
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
    bindings: &'a Bindings,
    nesting: Vec<NestingState>,
    call_site: Span,
}

fn expand_subtree(
    ctx: &mut ExpandCtx<'_>,
    template: &MetaTemplate,
    delimiter: Option<Delimiter<Span>>,
    arena: &mut Vec<tt::TokenTree<Span>>,
    marker: impl Fn(&mut Span) + Copy,
) -> ExpandResult<tt::Subtree<Span>> {
    // remember how many elements are in the arena now - when returning, we want to drain exactly how many elements we added. This way, the recursive uses of the arena get their own "view" of the arena, but will reuse the allocation
    let start_elements = arena.len();
    let mut err = None;
    'ops: for op in template.iter() {
        match op {
            Op::Literal(it) => arena.push(
                tt::Leaf::from({
                    let mut it = it.clone();
                    marker(&mut it.span);
                    it
                })
                .into(),
            ),
            Op::Ident(it) => arena.push(
                tt::Leaf::from({
                    let mut it = it.clone();
                    marker(&mut it.span);
                    it
                })
                .into(),
            ),
            Op::Punct(puncts) => {
                for punct in puncts.as_slice() {
                    arena.push(
                        tt::Leaf::from({
                            let mut it = *punct;
                            marker(&mut it.span);
                            it
                        })
                        .into(),
                    );
                }
            }
            Op::Subtree { tokens, delimiter } => {
                let mut delimiter = *delimiter;
                marker(&mut delimiter.open);
                marker(&mut delimiter.close);
                let ExpandResult { value: tt, err: e } =
                    expand_subtree(ctx, tokens, Some(delimiter), arena, marker);
                err = err.or(e);
                arena.push(tt.into());
            }
            Op::Var { name, id, .. } => {
                let ExpandResult { value: fragment, err: e } = expand_var(ctx, name, *id, marker);
                err = err.or(e);
                push_fragment(ctx, arena, fragment);
            }
            Op::Repeat { tokens: subtree, kind, separator } => {
                let ExpandResult { value: fragment, err: e } =
                    expand_repeat(ctx, subtree, *kind, separator.as_deref(), arena, marker);
                err = err.or(e);
                push_fragment(ctx, arena, fragment)
            }
            Op::Ignore { name, id } => {
                // Expand the variable, but ignore the result. This registers the repetition count.
                // FIXME: Any emitted errors are dropped.
                expand_var(ctx, name, *id, marker);
            }
            Op::Index { depth } => {
                let index =
                    ctx.nesting.get(ctx.nesting.len() - 1 - depth).map_or(0, |nest| nest.idx);
                arena.push(
                    tt::Leaf::Literal(tt::Literal {
                        symbol: Symbol::integer(index),
                        span: ctx.call_site,
                        kind: tt::LitKind::Integer,
                        suffix: None,
                    })
                    .into(),
                );
            }
            Op::Len { depth } => {
                let length = ctx.nesting.get(ctx.nesting.len() - 1 - depth).map_or(0, |_nest| {
                    // FIXME: to be implemented
                    0
                });
                arena.push(
                    tt::Leaf::Literal(tt::Literal {
                        symbol: Symbol::integer(length),
                        span: ctx.call_site,
                        kind: tt::LitKind::Integer,
                        suffix: None,
                    })
                    .into(),
                );
            }
            Op::Count { name, depth } => {
                let mut binding = match ctx.bindings.get(name, ctx.call_site) {
                    Ok(b) => b,
                    Err(e) => {
                        if err.is_none() {
                            err = Some(e);
                        }
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

                arena.push(
                    tt::Leaf::Literal(tt::Literal {
                        symbol: Symbol::integer(res),
                        span: ctx.call_site,
                        suffix: None,
                        kind: tt::LitKind::Integer,
                    })
                    .into(),
                );
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
                            let value = match &var_value {
                                Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Ident(ident))) => {
                                    ident.sym.as_str()
                                }
                                Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Literal(lit))) => {
                                    lit.symbol.as_str()
                                }
                                _ => {
                                    if err.is_none() {
                                        err = Some(ExpandError::binding_error(var.span, "metavariables of `${concat(..)}` must be of type `ident`, `literal` or `tt`"))
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
                arena.push(tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                    is_raw,
                    span: result_span,
                    sym: Symbol::intern(&concatenated),
                })));
            }
        }
    }
    // drain the elements added in this instance of expand_subtree
    let tts = arena.drain(start_elements..).collect();
    ExpandResult {
        value: tt::Subtree {
            delimiter: delimiter.unwrap_or_else(|| tt::Delimiter::invisible_spanned(ctx.call_site)),
            token_trees: tts,
        },
        err,
    }
}

fn expand_var(
    ctx: &mut ExpandCtx<'_>,
    v: &Symbol,
    id: Span,
    marker: impl Fn(&mut Span),
) -> ExpandResult<Fragment> {
    // We already handle $crate case in mbe parser
    debug_assert!(*v != sym::crate_);

    match ctx.bindings.get_fragment(v, id, &mut ctx.nesting, marker) {
        Ok(it) => ExpandResult::ok(it),
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
            let tt = tt::Subtree {
                delimiter: tt::Delimiter::invisible_spanned(id),
                token_trees: Box::new([
                    tt::Leaf::from(tt::Punct { char: '$', spacing: tt::Spacing::Alone, span: id })
                        .into(),
                    tt::Leaf::from(tt::Ident {
                        sym: v.clone(),
                        span: id,
                        is_raw: tt::IdentIsRaw::No,
                    })
                    .into(),
                ]),
            }
            .into();
            ExpandResult::ok(Fragment::Tokens(tt))
        }
        Err(e) => ExpandResult {
            value: Fragment::Tokens(tt::TokenTree::Subtree(tt::Subtree::empty(tt::DelimSpan {
                open: ctx.call_site,
                close: ctx.call_site,
            }))),
            err: Some(e),
        },
    }
}

fn expand_repeat(
    ctx: &mut ExpandCtx<'_>,
    template: &MetaTemplate,
    kind: RepeatKind,
    separator: Option<&Separator>,
    arena: &mut Vec<tt::TokenTree<Span>>,
    marker: impl Fn(&mut Span) + Copy,
) -> ExpandResult<Fragment> {
    let mut buf: Vec<tt::TokenTree<Span>> = Vec::new();
    ctx.nesting.push(NestingState { idx: 0, at_end: false, hit: false });
    // Dirty hack to make macro-expansion terminate.
    // This should be replaced by a proper macro-by-example implementation
    let limit = 65536;
    let mut has_seps = 0;
    let mut counter = 0;
    let mut err = None;

    loop {
        let ExpandResult { value: mut t, err: e } =
            expand_subtree(ctx, template, None, arena, marker);
        let nesting_state = ctx.nesting.last_mut().unwrap();
        if nesting_state.at_end || !nesting_state.hit {
            break;
        }
        nesting_state.idx += 1;
        nesting_state.hit = false;

        counter += 1;
        if counter == limit {
            tracing::warn!(
                "expand_tt in repeat pattern exceed limit => {:#?}\n{:#?}",
                template,
                ctx
            );
            return ExpandResult {
                value: Fragment::Tokens(
                    tt::Subtree {
                        delimiter: tt::Delimiter::invisible_spanned(ctx.call_site),
                        token_trees: Box::new([]),
                    }
                    .into(),
                ),
                err: Some(ExpandError::new(ctx.call_site, ExpandErrorKind::LimitExceeded)),
            };
        }

        if e.is_some() {
            err = err.or(e);
            continue;
        }

        t.delimiter.kind = tt::DelimiterKind::Invisible;
        push_subtree(&mut buf, t);

        if let Some(sep) = separator {
            has_seps = match sep {
                Separator::Ident(ident) => {
                    buf.push(tt::Leaf::from(ident.clone()).into());
                    1
                }
                Separator::Literal(lit) => {
                    buf.push(tt::Leaf::from(lit.clone()).into());
                    1
                }
                Separator::Puncts(puncts) => {
                    for &punct in puncts {
                        buf.push(tt::Leaf::from(punct).into());
                    }
                    puncts.len()
                }
            };
        }

        if RepeatKind::ZeroOrOne == kind {
            break;
        }
    }

    ctx.nesting.pop().unwrap();
    for _ in 0..has_seps {
        buf.pop();
    }

    // Check if it is a single token subtree without any delimiter
    // e.g {Delimiter:None> ['>'] /Delimiter:None>}
    let tt = tt::Subtree {
        delimiter: tt::Delimiter::invisible_spanned(ctx.call_site),
        token_trees: buf.into_boxed_slice(),
    };

    if RepeatKind::OneOrMore == kind && counter == 0 {
        let span = tt.delimiter.open;
        return ExpandResult {
            value: Fragment::Tokens(tt.into()),
            err: Some(ExpandError::new(span, ExpandErrorKind::UnexpectedToken)),
        };
    }
    ExpandResult { value: Fragment::Tokens(tt.into()), err }
}

fn push_fragment(ctx: &ExpandCtx<'_>, buf: &mut Vec<tt::TokenTree<Span>>, fragment: Fragment) {
    match fragment {
        Fragment::Tokens(tt::TokenTree::Subtree(tt)) => push_subtree(buf, tt),
        Fragment::Expr(sub) => {
            push_subtree(buf, sub);
        }
        Fragment::Path(tt) => fix_up_and_push_path_tt(ctx, buf, tt),
        Fragment::Tokens(tt) => buf.push(tt),
        Fragment::Empty => (),
    }
}

fn push_subtree(buf: &mut Vec<tt::TokenTree<Span>>, tt: tt::Subtree<Span>) {
    match tt.delimiter.kind {
        tt::DelimiterKind::Invisible => buf.extend(Vec::from(tt.token_trees)),
        _ => buf.push(tt.into()),
    }
}

/// Inserts the path separator `::` between an identifier and its following generic
/// argument list, and then pushes into the buffer. See [`Fragment::Path`] for why
/// we need this fixup.
fn fix_up_and_push_path_tt(
    ctx: &ExpandCtx<'_>,
    buf: &mut Vec<tt::TokenTree<Span>>,
    subtree: tt::Subtree<Span>,
) {
    stdx::always!(matches!(subtree.delimiter.kind, tt::DelimiterKind::Invisible));
    let mut prev_was_ident = false;
    // Note that we only need to fix up the top-level `TokenTree`s because the
    // context of the paths in the descendant `Subtree`s won't be changed by the
    // mbe transcription.
    for tt in Vec::from(subtree.token_trees) {
        if prev_was_ident {
            // Pedantically, `(T) -> U` in `FnOnce(T) -> U` is treated as a generic
            // argument list and thus needs `::` between it and `FnOnce`. However in
            // today's Rust this type of path *semantically* cannot appear as a
            // top-level expression-context path, so we can safely ignore it.
            if let tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct { char: '<', .. })) = tt {
                buf.push(
                    tt::Leaf::Punct(tt::Punct {
                        char: ':',
                        spacing: tt::Spacing::Joint,
                        span: ctx.call_site,
                    })
                    .into(),
                );
                buf.push(
                    tt::Leaf::Punct(tt::Punct {
                        char: ':',
                        spacing: tt::Spacing::Alone,
                        span: ctx.call_site,
                    })
                    .into(),
                );
            }
        }
        prev_was_ident = matches!(tt, tt::TokenTree::Leaf(tt::Leaf::Ident(_)));
        buf.push(tt);
    }
}

/// Handles `${count(t, depth)}`. `our_depth` is the recursion depth and `count_depth` is the depth
/// defined by the metavar expression.
fn count(binding: &Binding, depth_curr: usize, depth_max: usize) -> usize {
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
