//! Transcriber takes a template, like `fn $ident() {}`, a set of bindings like
//! `$ident => foo`, interpolates variables in the template, to get `fn foo() {}`

use syntax::SmolStr;
use tt::{Delimiter, Span};

use crate::{
    expander::{Binding, Bindings, Fragment},
    parser::{MetaVarKind, Op, RepeatKind, Separator},
    CountError, ExpandError, ExpandResult, MetaTemplate,
};

impl<S: Span> Bindings<S> {
    fn get(&self, name: &str) -> Result<&Binding<S>, ExpandError> {
        match self.inner.get(name) {
            Some(binding) => Ok(binding),
            None => Err(ExpandError::UnresolvedBinding(Box::new(Box::from(name)))),
        }
    }

    fn get_fragment(
        &self,
        name: &str,
        mut span: S,
        nesting: &mut [NestingState],
        marker: impl Fn(&mut S),
    ) -> Result<Fragment<S>, ExpandError> {
        macro_rules! binding_err {
            ($($arg:tt)*) => { ExpandError::binding_error(format!($($arg)*)) };
        }

        let mut b = self.get(name)?;
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
                    | MetaVarKind::Expr
                    | MetaVarKind::Ident => {
                        Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                            text: SmolStr::new_static("missing"),
                            span,
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
                                    text: SmolStr::new_static("missing"),
                                    span,
                                })),
                            ]),
                        }))
                    }
                    MetaVarKind::Literal => {
                        Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                            text: SmolStr::new_static("\"missing\""),
                            span,
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

pub(super) fn transcribe<S: Span>(
    template: &MetaTemplate<S>,
    bindings: &Bindings<S>,
    marker: impl Fn(&mut S) + Copy,
    new_meta_vars: bool,
    call_site: S,
) -> ExpandResult<tt::Subtree<S>> {
    let mut ctx = ExpandCtx { bindings, nesting: Vec::new(), new_meta_vars, call_site };
    let mut arena: Vec<tt::TokenTree<S>> = Vec::new();
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
struct ExpandCtx<'a, S> {
    bindings: &'a Bindings<S>,
    nesting: Vec<NestingState>,
    new_meta_vars: bool,
    call_site: S,
}

fn expand_subtree<S: Span>(
    ctx: &mut ExpandCtx<'_, S>,
    template: &MetaTemplate<S>,
    delimiter: Option<Delimiter<S>>,
    arena: &mut Vec<tt::TokenTree<S>>,
    marker: impl Fn(&mut S) + Copy,
) -> ExpandResult<tt::Subtree<S>> {
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
                for punct in puncts {
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
                    expand_repeat(ctx, subtree, *kind, separator, arena, marker);
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
                        text: index.to_string().into(),
                        span: ctx.call_site,
                    })
                    .into(),
                );
            }
            Op::Length { depth } => {
                let length = ctx.nesting.get(ctx.nesting.len() - 1 - depth).map_or(0, |_nest| {
                    // FIXME: to be implemented
                    0
                });
                arena.push(
                    tt::Leaf::Literal(tt::Literal {
                        text: length.to_string().into(),
                        span: ctx.call_site,
                    })
                    .into(),
                );
            }
            Op::Count { name, depth } => {
                let mut binding = match ctx.bindings.get(name.as_str()) {
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

                let res = if ctx.new_meta_vars {
                    count(binding, 0, depth.unwrap_or(0))
                } else {
                    count_old(binding, 0, *depth)
                };

                let c = match res {
                    Ok(c) => c,
                    Err(e) => {
                        // XXX: It *might* make sense to emit a dummy integer value like `0` here.
                        // That would type inference a bit more robust in cases like
                        // `v[${count(t)}]` where index doesn't matter, but also could lead to
                        // wrong infefrence for cases like `tup.${count(t)}` where index itself
                        // does matter.
                        if err.is_none() {
                            err = Some(e.into());
                        }
                        continue;
                    }
                };
                arena.push(
                    tt::Leaf::Literal(tt::Literal {
                        text: c.to_string().into(),
                        span: ctx.call_site,
                    })
                    .into(),
                );
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

fn expand_var<S: Span>(
    ctx: &mut ExpandCtx<'_, S>,
    v: &SmolStr,
    id: S,
    marker: impl Fn(&mut S),
) -> ExpandResult<Fragment<S>> {
    // We already handle $crate case in mbe parser
    debug_assert!(v != "crate");

    match ctx.bindings.get_fragment(v, id, &mut ctx.nesting, marker) {
        Ok(it) => ExpandResult::ok(it),
        Err(ExpandError::UnresolvedBinding(_)) => {
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
                    tt::Leaf::from(tt::Ident { text: v.clone(), span: id }).into(),
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

fn expand_repeat<S: Span>(
    ctx: &mut ExpandCtx<'_, S>,
    template: &MetaTemplate<S>,
    kind: RepeatKind,
    separator: &Option<Separator<S>>,
    arena: &mut Vec<tt::TokenTree<S>>,
    marker: impl Fn(&mut S) + Copy,
) -> ExpandResult<Fragment<S>> {
    let mut buf: Vec<tt::TokenTree<S>> = Vec::new();
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
                err: Some(ExpandError::LimitExceeded),
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
    }
    .into();

    if RepeatKind::OneOrMore == kind && counter == 0 {
        return ExpandResult {
            value: Fragment::Tokens(tt),
            err: Some(ExpandError::UnexpectedToken),
        };
    }
    ExpandResult { value: Fragment::Tokens(tt), err }
}

fn push_fragment<S: Span>(
    ctx: &ExpandCtx<'_, S>,
    buf: &mut Vec<tt::TokenTree<S>>,
    fragment: Fragment<S>,
) {
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

fn push_subtree<S>(buf: &mut Vec<tt::TokenTree<S>>, tt: tt::Subtree<S>) {
    match tt.delimiter.kind {
        tt::DelimiterKind::Invisible => buf.extend(Vec::from(tt.token_trees)),
        _ => buf.push(tt.into()),
    }
}

/// Inserts the path separator `::` between an identifier and its following generic
/// argument list, and then pushes into the buffer. See [`Fragment::Path`] for why
/// we need this fixup.
fn fix_up_and_push_path_tt<S: Span>(
    ctx: &ExpandCtx<'_, S>,
    buf: &mut Vec<tt::TokenTree<S>>,
    subtree: tt::Subtree<S>,
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
fn count<S>(
    binding: &Binding<S>,
    depth_curr: usize,
    depth_max: usize,
) -> Result<usize, CountError> {
    match binding {
        Binding::Nested(bs) => {
            if depth_curr == depth_max {
                Ok(bs.len())
            } else {
                bs.iter().map(|b| count(b, depth_curr + 1, depth_max)).sum()
            }
        }
        Binding::Empty => Ok(0),
        Binding::Fragment(_) | Binding::Missing(_) => Ok(1),
    }
}

fn count_old<S>(
    binding: &Binding<S>,
    our_depth: usize,
    count_depth: Option<usize>,
) -> Result<usize, CountError> {
    match binding {
        Binding::Nested(bs) => match count_depth {
            None => bs.iter().map(|b| count_old(b, our_depth + 1, None)).sum(),
            Some(0) => Ok(bs.len()),
            Some(d) => bs.iter().map(|b| count_old(b, our_depth + 1, Some(d - 1))).sum(),
        },
        Binding::Empty => Ok(0),
        Binding::Fragment(_) | Binding::Missing(_) => {
            if our_depth == 0 {
                // `${count(t)}` is placed inside the innermost repetition. This includes cases
                // where `t` is not a repeated fragment.
                Err(CountError::Misplaced)
            } else if count_depth.is_none() {
                Ok(1)
            } else {
                // We've reached at the innermost repeated fragment, but the user wants us to go
                // further!
                Err(CountError::OutOfBounds)
            }
        }
    }
}
