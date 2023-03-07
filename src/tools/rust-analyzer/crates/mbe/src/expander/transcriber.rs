//! Transcriber takes a template, like `fn $ident() {}`, a set of bindings like
//! `$ident => foo`, interpolates variables in the template, to get `fn foo() {}`

use syntax::SmolStr;

use crate::{
    expander::{Binding, Bindings, Fragment},
    parser::{MetaVarKind, Op, RepeatKind, Separator},
    tt::{self, Delimiter},
    ExpandError, ExpandResult, MetaTemplate,
};

impl Bindings {
    fn contains(&self, name: &str) -> bool {
        self.inner.contains_key(name)
    }

    fn get(&self, name: &str, nesting: &mut [NestingState]) -> Result<Fragment, ExpandError> {
        macro_rules! binding_err {
            ($($arg:tt)*) => { ExpandError::binding_error(format!($($arg)*)) };
        }

        let mut b: &Binding =
            self.inner.get(name).ok_or_else(|| binding_err!("could not find binding `{name}`"))?;
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
            Binding::Fragment(it) => Ok(it.clone()),
            // emit some reasonable default expansion for missing bindings,
            // this gives better recovery than emitting the `$fragment-name` verbatim
            Binding::Missing(it) => Ok(match it {
                MetaVarKind::Stmt => {
                    Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Punct(tt::Punct {
                        span: tt::TokenId::unspecified(),
                        char: ';',
                        spacing: tt::Spacing::Alone,
                    })))
                }
                MetaVarKind::Block => Fragment::Tokens(tt::TokenTree::Subtree(tt::Subtree {
                    delimiter: tt::Delimiter {
                        open: tt::TokenId::unspecified(),
                        close: tt::TokenId::unspecified(),
                        kind: tt::DelimiterKind::Brace,
                    },
                    token_trees: vec![],
                })),
                // FIXME: Meta and Item should get proper defaults
                MetaVarKind::Meta | MetaVarKind::Item | MetaVarKind::Tt | MetaVarKind::Vis => {
                    Fragment::Tokens(tt::TokenTree::Subtree(tt::Subtree {
                        delimiter: tt::Delimiter::UNSPECIFIED,
                        token_trees: vec![],
                    }))
                }
                MetaVarKind::Path
                | MetaVarKind::Ty
                | MetaVarKind::Pat
                | MetaVarKind::PatParam
                | MetaVarKind::Expr
                | MetaVarKind::Ident => {
                    Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                        text: SmolStr::new_inline("missing"),
                        span: tt::TokenId::unspecified(),
                    })))
                }
                MetaVarKind::Lifetime => {
                    Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                        text: SmolStr::new_inline("'missing"),
                        span: tt::TokenId::unspecified(),
                    })))
                }
                MetaVarKind::Literal => {
                    Fragment::Tokens(tt::TokenTree::Leaf(tt::Leaf::Ident(tt::Ident {
                        text: SmolStr::new_inline("\"missing\""),
                        span: tt::TokenId::unspecified(),
                    })))
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
) -> ExpandResult<tt::Subtree> {
    let mut ctx = ExpandCtx { bindings, nesting: Vec::new() };
    let mut arena: Vec<tt::TokenTree> = Vec::new();
    expand_subtree(&mut ctx, template, None, &mut arena)
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
}

fn expand_subtree(
    ctx: &mut ExpandCtx<'_>,
    template: &MetaTemplate,
    delimiter: Option<Delimiter>,
    arena: &mut Vec<tt::TokenTree>,
) -> ExpandResult<tt::Subtree> {
    // remember how many elements are in the arena now - when returning, we want to drain exactly how many elements we added. This way, the recursive uses of the arena get their own "view" of the arena, but will reuse the allocation
    let start_elements = arena.len();
    let mut err = None;
    for op in template.iter() {
        match op {
            Op::Literal(it) => arena.push(tt::Leaf::from(it.clone()).into()),
            Op::Ident(it) => arena.push(tt::Leaf::from(it.clone()).into()),
            Op::Punct(puncts) => {
                for punct in puncts {
                    arena.push(tt::Leaf::from(*punct).into());
                }
            }
            Op::Subtree { tokens, delimiter } => {
                let ExpandResult { value: tt, err: e } =
                    expand_subtree(ctx, tokens, Some(*delimiter), arena);
                err = err.or(e);
                arena.push(tt.into());
            }
            Op::Var { name, id, .. } => {
                let ExpandResult { value: fragment, err: e } = expand_var(ctx, name, *id);
                err = err.or(e);
                push_fragment(arena, fragment);
            }
            Op::Repeat { tokens: subtree, kind, separator } => {
                let ExpandResult { value: fragment, err: e } =
                    expand_repeat(ctx, subtree, *kind, separator, arena);
                err = err.or(e);
                push_fragment(arena, fragment)
            }
            Op::Ignore { name, id } => {
                // Expand the variable, but ignore the result. This registers the repetition count.
                expand_var(ctx, name, *id);
            }
            Op::Index { depth } => {
                let index = ctx
                    .nesting
                    .get(ctx.nesting.len() - 1 - (*depth as usize))
                    .map_or(0, |nest| nest.idx);
                arena.push(
                    tt::Leaf::Literal(tt::Literal {
                        text: index.to_string().into(),
                        span: tt::TokenId::unspecified(),
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
            delimiter: delimiter.unwrap_or_else(tt::Delimiter::unspecified),
            token_trees: tts,
        },
        err,
    }
}

fn expand_var(ctx: &mut ExpandCtx<'_>, v: &SmolStr, id: tt::TokenId) -> ExpandResult<Fragment> {
    // We already handle $crate case in mbe parser
    debug_assert!(v != "crate");

    if !ctx.bindings.contains(v) {
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
            delimiter: tt::Delimiter::UNSPECIFIED,
            token_trees: vec![
                tt::Leaf::from(tt::Punct { char: '$', spacing: tt::Spacing::Alone, span: id })
                    .into(),
                tt::Leaf::from(tt::Ident { text: v.clone(), span: id }).into(),
            ],
        }
        .into();
        ExpandResult::ok(Fragment::Tokens(tt))
    } else {
        ctx.bindings.get(v, &mut ctx.nesting).map_or_else(
            |e| ExpandResult {
                value: Fragment::Tokens(tt::TokenTree::Subtree(tt::Subtree {
                    delimiter: tt::Delimiter::unspecified(),
                    token_trees: vec![],
                })),
                err: Some(e),
            },
            ExpandResult::ok,
        )
    }
}

fn expand_repeat(
    ctx: &mut ExpandCtx<'_>,
    template: &MetaTemplate,
    kind: RepeatKind,
    separator: &Option<Separator>,
    arena: &mut Vec<tt::TokenTree>,
) -> ExpandResult<Fragment> {
    let mut buf: Vec<tt::TokenTree> = Vec::new();
    ctx.nesting.push(NestingState { idx: 0, at_end: false, hit: false });
    // Dirty hack to make macro-expansion terminate.
    // This should be replaced by a proper macro-by-example implementation
    let limit = 65536;
    let mut has_seps = 0;
    let mut counter = 0;

    loop {
        let ExpandResult { value: mut t, err: e } = expand_subtree(ctx, template, None, arena);
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
                    tt::Subtree { delimiter: tt::Delimiter::unspecified(), token_trees: vec![] }
                        .into(),
                ),
                err: Some(ExpandError::LimitExceeded),
            };
        }

        if e.is_some() {
            continue;
        }

        t.delimiter = tt::Delimiter::unspecified();
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
    let tt = tt::Subtree { delimiter: tt::Delimiter::unspecified(), token_trees: buf }.into();

    if RepeatKind::OneOrMore == kind && counter == 0 {
        return ExpandResult {
            value: Fragment::Tokens(tt),
            err: Some(ExpandError::UnexpectedToken),
        };
    }
    ExpandResult::ok(Fragment::Tokens(tt))
}

fn push_fragment(buf: &mut Vec<tt::TokenTree>, fragment: Fragment) {
    match fragment {
        Fragment::Tokens(tt::TokenTree::Subtree(tt)) => push_subtree(buf, tt),
        Fragment::Expr(tt::TokenTree::Subtree(mut tt)) => {
            if tt.delimiter.kind == tt::DelimiterKind::Invisible {
                tt.delimiter = tt::Delimiter {
                    open: tt::TokenId::UNSPECIFIED,
                    close: tt::TokenId::UNSPECIFIED,
                    kind: tt::DelimiterKind::Parenthesis,
                };
            }
            buf.push(tt.into())
        }
        Fragment::Tokens(tt) | Fragment::Expr(tt) => buf.push(tt),
    }
}

fn push_subtree(buf: &mut Vec<tt::TokenTree>, tt: tt::Subtree) {
    match tt.delimiter.kind {
        tt::DelimiterKind::Invisible => buf.extend(tt.token_trees),
        _ => buf.push(tt.into()),
    }
}
