//! Transcriber takes a template, like `fn $ident() {}`, a set of bindings like
//! `$ident => foo`, interpolates variables in the template, to get `fn foo() {}`

use syntax::SmolStr;
use tt::{Delimiter, Subtree};

use crate::{
    expander::{Binding, Bindings, Fragment},
    parser::{Op, RepeatKind, Separator},
    ExpandError, ExpandResult, MetaTemplate,
};

impl Bindings {
    fn contains(&self, name: &str) -> bool {
        self.inner.contains_key(name)
    }

    fn get(&self, name: &str, nesting: &mut [NestingState]) -> Result<&Fragment, ExpandError> {
        macro_rules! binding_err {
            ($($arg:tt)*) => { ExpandError::BindingError(format!($($arg)*)) };
        }

        let mut b: &Binding = self
            .inner
            .get(name)
            .ok_or_else(|| binding_err!("could not find binding `{}`", name))?;
        for nesting_state in nesting.iter_mut() {
            nesting_state.hit = true;
            b = match b {
                Binding::Fragment(_) => break,
                Binding::Nested(bs) => bs.get(nesting_state.idx).ok_or_else(|| {
                    nesting_state.at_end = true;
                    binding_err!("could not find nested binding `{}`", name)
                })?,
                Binding::Empty => {
                    nesting_state.at_end = true;
                    return Err(binding_err!("could not find empty binding `{}`", name));
                }
            };
        }
        match b {
            Binding::Fragment(it) => Ok(it),
            Binding::Nested(_) => {
                Err(binding_err!("expected simple binding, found nested binding `{}`", name))
            }
            Binding::Empty => {
                Err(binding_err!("expected simple binding, found empty binding `{}`", name))
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
    ctx: &mut ExpandCtx,
    template: &MetaTemplate,
    delimiter: Option<Delimiter>,
    arena: &mut Vec<tt::TokenTree>,
) -> ExpandResult<tt::Subtree> {
    // remember how many elements are in the arena now - when returning, we want to drain exactly how many elements we added. This way, the recursive uses of the arena get their own "view" of the arena, but will reuse the allocation
    let start_elements = arena.len();
    let mut err = None;
    for op in template.iter() {
        match op {
            Op::Leaf(tt) => arena.push(tt.clone().into()),
            Op::Subtree { tokens, delimiter } => {
                let ExpandResult { value: tt, err: e } =
                    expand_subtree(ctx, tokens, *delimiter, arena);
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
        }
    }
    // drain the elements added in this instance of expand_subtree
    let tts = arena.drain(start_elements..).collect();
    ExpandResult { value: tt::Subtree { delimiter, token_trees: tts }, err }
}

fn expand_var(ctx: &mut ExpandCtx, v: &SmolStr, id: tt::TokenId) -> ExpandResult<Fragment> {
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
            delimiter: None,
            token_trees: vec![
                tt::Leaf::from(tt::Punct { char: '$', spacing: tt::Spacing::Alone, id }).into(),
                tt::Leaf::from(tt::Ident { text: v.clone(), id }).into(),
            ],
        }
        .into();
        ExpandResult::ok(Fragment::Tokens(tt))
    } else {
        ctx.bindings.get(v, &mut ctx.nesting).map_or_else(
            |e| ExpandResult { value: Fragment::Tokens(tt::TokenTree::empty()), err: Some(e) },
            |b| ExpandResult::ok(b.clone()),
        )
    }
}

fn expand_repeat(
    ctx: &mut ExpandCtx,
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
                value: Fragment::Tokens(Subtree::default().into()),
                err: Some(ExpandError::Other("Expand exceed limit".to_string())),
            };
        }

        if e.is_some() {
            continue;
        }

        t.delimiter = None;
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
    let tt = tt::Subtree { delimiter: None, token_trees: buf }.into();

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
            if tt.delimiter.is_none() {
                tt.delimiter = Some(tt::Delimiter {
                    id: tt::TokenId::unspecified(),
                    kind: tt::DelimiterKind::Parenthesis,
                })
            }
            buf.push(tt.into())
        }
        Fragment::Tokens(tt) | Fragment::Expr(tt) => buf.push(tt),
    }
}

fn push_subtree(buf: &mut Vec<tt::TokenTree>, tt: tt::Subtree) {
    match tt.delimiter {
        None => buf.extend(tt.token_trees),
        Some(_) => buf.push(tt.into()),
    }
}
