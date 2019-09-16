use ra_syntax::SmolStr;

use crate::{
    mbe_expander::{Binding, Bindings, Fragment},
    ExpandError,
};

impl Bindings {
    fn contains(&self, name: &SmolStr) -> bool {
        self.inner.contains_key(name)
    }

    fn get(&self, name: &SmolStr, nesting: &[usize]) -> Result<&Fragment, ExpandError> {
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
    bindings: &Bindings,
    template: &crate::Subtree,
) -> Result<tt::Subtree, ExpandError> {
    let mut ctx = ExpandCtx { bindings: &bindings, nesting: Vec::new(), var_expanded: false };
    expand_subtree(template, &mut ctx)
}

#[derive(Debug)]
struct ExpandCtx<'a> {
    bindings: &'a Bindings,
    nesting: Vec<usize>,
    var_expanded: bool,
}

fn expand_subtree(
    template: &crate::Subtree,
    ctx: &mut ExpandCtx,
) -> Result<tt::Subtree, ExpandError> {
    let mut buf: Vec<tt::TokenTree> = Vec::new();
    for tt in template.token_trees.iter() {
        let tt = expand_tt(tt, ctx)?;
        push_fragment(&mut buf, tt);
    }

    Ok(tt::Subtree { delimiter: template.delimiter, token_trees: buf })
}

fn expand_tt(template: &crate::TokenTree, ctx: &mut ExpandCtx) -> Result<Fragment, ExpandError> {
    let res: tt::TokenTree = match template {
        crate::TokenTree::Subtree(subtree) => expand_subtree(subtree, ctx)?.into(),
        crate::TokenTree::Repeat(repeat) => {
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

            while let Ok(t) = expand_subtree(&repeat.subtree, ctx) {
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

                if let Some(ref sep) = repeat.separator {
                    match sep {
                        crate::Separator::Ident(ident) => {
                            has_seps = 1;
                            buf.push(tt::Leaf::from(ident.clone()).into());
                        }
                        crate::Separator::Literal(lit) => {
                            has_seps = 1;
                            buf.push(tt::Leaf::from(lit.clone()).into());
                        }

                        crate::Separator::Puncts(puncts) => {
                            has_seps = puncts.len();
                            for punct in puncts {
                                buf.push(tt::Leaf::from(*punct).into());
                            }
                        }
                    }
                }

                if let crate::RepeatKind::ZeroOrOne = repeat.kind {
                    break;
                }
            }

            // Restore the `var_expanded` by combining old one and the new one
            ctx.var_expanded = some_var_expanded || old_var_expanded;

            ctx.nesting.pop().unwrap();
            for _ in 0..has_seps {
                buf.pop();
            }

            if crate::RepeatKind::OneOrMore == repeat.kind && counter == 0 {
                return Err(ExpandError::UnexpectedToken);
            }

            // Check if it is a single token subtree without any delimiter
            // e.g {Delimiter:None> ['>'] /Delimiter:None>}
            tt::Subtree { delimiter: tt::Delimiter::None, token_trees: buf }.into()
        }
        crate::TokenTree::Leaf(leaf) => match leaf {
            crate::Leaf::Ident(ident) => tt::Leaf::from(tt::Ident {
                text: ident.text.clone(),
                id: tt::TokenId::unspecified(),
            })
            .into(),
            crate::Leaf::Punct(punct) => tt::Leaf::from(*punct).into(),
            crate::Leaf::Var(v) => {
                if v.text == "crate" {
                    // FIXME: Properly handle $crate token
                    tt::Leaf::from(tt::Ident {
                        text: "$crate".into(),
                        id: tt::TokenId::unspecified(),
                    })
                    .into()
                } else if !ctx.bindings.contains(&v.text) {
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
                    tt::Subtree {
                        delimiter: tt::Delimiter::None,
                        token_trees: vec![
                            tt::Leaf::from(tt::Punct { char: '$', spacing: tt::Spacing::Alone })
                                .into(),
                            tt::Leaf::from(tt::Ident {
                                text: v.text.clone(),
                                id: tt::TokenId::unspecified(),
                            })
                            .into(),
                        ],
                    }
                    .into()
                } else {
                    let fragment = ctx.bindings.get(&v.text, &ctx.nesting)?.clone();
                    ctx.var_expanded = true;
                    return Ok(fragment);
                }
            }
            crate::Leaf::Literal(l) => tt::Leaf::from(tt::Literal { text: l.text.clone() }).into(),
        },
    };
    Ok(Fragment::Tokens(res))
}

fn push_fragment(buf: &mut Vec<tt::TokenTree>, fragment: Fragment) {
    match fragment {
        Fragment::Tokens(tt::TokenTree::Subtree(tt)) => push_subtree(buf, tt),
        Fragment::Tokens(tt) | Fragment::Ast(tt) => buf.push(tt),
    }
}

fn push_subtree(buf: &mut Vec<tt::TokenTree>, tt: tt::Subtree) {
    match tt.delimiter {
        tt::Delimiter::None => buf.extend(tt.token_trees),
        _ => buf.push(tt.into()),
    }
}
