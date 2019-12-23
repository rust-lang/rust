//! Transcraber takes a template, like `fn $ident() {}`, a set of bindings like
//! `$ident => foo`, interpolates variables in the template, to get `fn foo() {}`

use ra_syntax::SmolStr;

use crate::{
    mbe_expander::{Binding, Bindings, Fragment},
    parser::{parse_template, Op, RepeatKind, Separator},
    ExpandError,
};

impl Bindings {
    fn contains(&self, name: &str) -> bool {
        self.inner.contains_key(name)
    }

    fn get(&self, name: &str, nesting: &[usize]) -> Result<&Fragment, ExpandError> {
        let mut b = self.inner.get(name).ok_or_else(|| {
            ExpandError::BindingError(format!("could not find binding `{}`", name))
        })?;
        for &idx in nesting.iter() {
            b = match b {
                Binding::Fragment(_) => break,
                Binding::Nested(bs) => bs.get(idx).ok_or_else(|| {
                    ExpandError::BindingError(format!("could not find nested binding `{}`", name))
                })?,
                Binding::Empty => {
                    return Err(ExpandError::BindingError(format!(
                        "could not find empty binding `{}`",
                        name
                    )))
                }
            };
        }
        match b {
            Binding::Fragment(it) => Ok(it),
            Binding::Nested(_) => Err(ExpandError::BindingError(format!(
                "expected simple binding, found nested binding `{}`",
                name
            ))),
            Binding::Empty => Err(ExpandError::BindingError(format!(
                "expected simple binding, found empty binding `{}`",
                name
            ))),
        }
    }
}

pub(super) fn transcribe(
    template: &tt::Subtree,
    bindings: &Bindings,
) -> Result<tt::Subtree, ExpandError> {
    assert!(template.delimiter == None);
    let mut ctx = ExpandCtx { bindings: &bindings, nesting: Vec::new(), var_expanded: false };
    expand_subtree(&mut ctx, template)
}

#[derive(Debug)]
struct ExpandCtx<'a> {
    bindings: &'a Bindings,
    nesting: Vec<usize>,
    var_expanded: bool,
}

fn expand_subtree(ctx: &mut ExpandCtx, template: &tt::Subtree) -> Result<tt::Subtree, ExpandError> {
    let mut buf: Vec<tt::TokenTree> = Vec::new();
    for op in parse_template(template) {
        match op? {
            Op::TokenTree(tt @ tt::TokenTree::Leaf(..)) => buf.push(tt.clone()),
            Op::TokenTree(tt::TokenTree::Subtree(tt)) => {
                let tt = expand_subtree(ctx, tt)?;
                buf.push(tt.into());
            }
            Op::Var { name, kind: _ } => {
                let fragment = expand_var(ctx, name)?;
                push_fragment(&mut buf, fragment);
            }
            Op::Repeat { subtree, kind, separator } => {
                let fragment = expand_repeat(ctx, subtree, kind, separator)?;
                push_fragment(&mut buf, fragment)
            }
        }
    }
    Ok(tt::Subtree { delimiter: template.delimiter, token_trees: buf })
}

fn expand_var(ctx: &mut ExpandCtx, v: &SmolStr) -> Result<Fragment, ExpandError> {
    let res = if v == "crate" {
        // We simply produce identifier `$crate` here. And it will be resolved when lowering ast to Path.
        let tt =
            tt::Leaf::from(tt::Ident { text: "$crate".into(), id: tt::TokenId::unspecified() })
                .into();
        Fragment::Tokens(tt)
    } else if !ctx.bindings.contains(v) {
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
                tt::Leaf::from(tt::Punct {
                    char: '$',
                    spacing: tt::Spacing::Alone,
                    id: tt::TokenId::unspecified(),
                })
                .into(),
                tt::Leaf::from(tt::Ident { text: v.clone(), id: tt::TokenId::unspecified() })
                    .into(),
            ],
        }
        .into();
        Fragment::Tokens(tt)
    } else {
        let fragment = ctx.bindings.get(&v, &ctx.nesting)?.clone();
        ctx.var_expanded = true;
        fragment
    };
    Ok(res)
}

fn expand_repeat(
    ctx: &mut ExpandCtx,
    template: &tt::Subtree,
    kind: RepeatKind,
    separator: Option<Separator>,
) -> Result<Fragment, ExpandError> {
    let mut buf: Vec<tt::TokenTree> = Vec::new();
    ctx.nesting.push(0);
    // Dirty hack to make macro-expansion terminate.
    // This should be replaced by a propper macro-by-example implementation
    let mut limit = 65536;
    let mut has_seps = 0;
    let mut counter = 0;

    // We store the old var expanded value, and restore it later
    // It is because before this `$repeat`,
    // it is possible some variables already expanad in the same subtree
    //
    // `some_var_expanded` keep check if the deeper subtree has expanded variables
    let mut some_var_expanded = false;
    let old_var_expanded = ctx.var_expanded;
    ctx.var_expanded = false;

    while let Ok(mut t) = expand_subtree(ctx, template) {
        t.delimiter = None;
        // if no var expanded in the child, we count it as a fail
        if !ctx.var_expanded {
            break;
        }

        // Reset `ctx.var_expandeded` to see if there is other expanded variable
        // in the next matching
        some_var_expanded = true;
        ctx.var_expanded = false;

        counter += 1;
        limit -= 1;
        if limit == 0 {
            log::warn!(
                "expand_tt excced in repeat pattern exceed limit => {:#?}\n{:#?}",
                template,
                ctx
            );
            break;
        }

        let idx = ctx.nesting.pop().unwrap();
        ctx.nesting.push(idx + 1);
        push_subtree(&mut buf, t);

        if let Some(ref sep) = separator {
            match sep {
                Separator::Ident(ident) => {
                    has_seps = 1;
                    buf.push(tt::Leaf::from(ident.clone()).into());
                }
                Separator::Literal(lit) => {
                    has_seps = 1;
                    buf.push(tt::Leaf::from(lit.clone()).into());
                }

                Separator::Puncts(puncts) => {
                    has_seps = puncts.len();
                    for punct in puncts {
                        buf.push(tt::Leaf::from(*punct).into());
                    }
                }
            }
        }

        if RepeatKind::ZeroOrOne == kind {
            break;
        }
    }

    // Restore the `var_expanded` by combining old one and the new one
    ctx.var_expanded = some_var_expanded || old_var_expanded;

    ctx.nesting.pop().unwrap();
    for _ in 0..has_seps {
        buf.pop();
    }

    if RepeatKind::OneOrMore == kind && counter == 0 {
        return Err(ExpandError::UnexpectedToken);
    }

    // Check if it is a single token subtree without any delimiter
    // e.g {Delimiter:None> ['>'] /Delimiter:None>}
    let tt = tt::Subtree { delimiter: None, token_trees: buf }.into();
    Ok(Fragment::Tokens(tt))
}

fn push_fragment(buf: &mut Vec<tt::TokenTree>, fragment: Fragment) {
    match fragment {
        Fragment::Tokens(tt::TokenTree::Subtree(tt)) => push_subtree(buf, tt),
        Fragment::Tokens(tt) | Fragment::Ast(tt) => buf.push(tt),
    }
}

fn push_subtree(buf: &mut Vec<tt::TokenTree>, tt: tt::Subtree) {
    match tt.delimiter {
        None => buf.extend(tt.token_trees),
        _ => buf.push(tt.into()),
    }
}
