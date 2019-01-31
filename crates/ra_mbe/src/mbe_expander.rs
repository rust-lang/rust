use rustc_hash::FxHashMap;
use ra_syntax::SmolStr;

use crate::tt_cursor::TtCursor;

pub fn exapnd(rules: &crate::MacroRules, input: &tt::Subtree) -> Option<tt::Subtree> {
    rules.rules.iter().find_map(|it| expand_rule(it, input))
}

fn expand_rule(rule: &crate::Rule, input: &tt::Subtree) -> Option<tt::Subtree> {
    let mut input = TtCursor::new(input);
    let bindings = match_lhs(&rule.lhs, &mut input)?;
    expand_subtree(&rule.rhs, &bindings, &mut Vec::new())
}

#[derive(Debug, Default)]
struct Bindings {
    inner: FxHashMap<SmolStr, Binding>,
}

#[derive(Debug)]
enum Binding {
    Simple(tt::TokenTree),
    Nested(Vec<Binding>),
}

impl Bindings {
    fn get(&self, name: &SmolStr, nesting: &[usize]) -> Option<&tt::TokenTree> {
        let mut b = self.inner.get(name)?;
        for &idx in nesting.iter() {
            b = match b {
                Binding::Simple(_) => break,
                Binding::Nested(bs) => bs.get(idx)?,
            };
        }
        match b {
            Binding::Simple(it) => Some(it),
            Binding::Nested(_) => None,
        }
    }
    fn push_nested(&mut self, nested: Bindings) -> Option<()> {
        for (key, value) in nested.inner {
            if !self.inner.contains_key(&key) {
                self.inner.insert(key.clone(), Binding::Nested(Vec::new()));
            }
            match self.inner.get_mut(&key) {
                Some(Binding::Nested(it)) => it.push(value),
                _ => return None,
            }
        }
        Some(())
    }
}

fn match_lhs(pattern: &crate::Subtree, input: &mut TtCursor) -> Option<Bindings> {
    let mut res = Bindings::default();
    for pat in pattern.token_trees.iter() {
        match pat {
            crate::TokenTree::Leaf(leaf) => match leaf {
                crate::Leaf::Var(crate::Var { text, kind }) => {
                    let kind = kind.clone()?;
                    match kind.as_str() {
                        "ident" => {
                            let ident = input.eat_ident()?.clone();
                            res.inner.insert(
                                text.clone(),
                                Binding::Simple(tt::Leaf::from(ident).into()),
                            );
                        }
                        _ => return None,
                    }
                }
                crate::Leaf::Punct(punct) => {
                    if input.eat_punct()? != punct {
                        return None;
                    }
                }
                _ => return None,
            },
            crate::TokenTree::Repeat(crate::Repeat {
                subtree,
                kind: _,
                separator,
            }) => {
                while let Some(nested) = match_lhs(subtree, input) {
                    res.push_nested(nested)?;
                    if separator.is_some() && !input.is_eof() {
                        input.eat_punct()?;
                    }
                }
            }
            _ => {}
        }
    }
    Some(res)
}

/*

macro_rules! impl_froms {
    ($e:ident: $($v:ident),*) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e {
                    $e::$v(it)
                }
            }
        )*
    }
}

impl_froms! (Foo: Bar, Baz)

*/

fn expand_subtree(
    template: &crate::Subtree,
    bindings: &Bindings,
    nesting: &mut Vec<usize>,
) -> Option<tt::Subtree> {
    let token_trees = template
        .token_trees
        .iter()
        .map(|it| expand_tt(it, bindings, nesting))
        .collect::<Option<Vec<_>>>()?;

    Some(tt::Subtree {
        token_trees,
        delimiter: template.delimiter,
    })
}

fn expand_tt(
    template: &crate::TokenTree,
    bindings: &Bindings,
    nesting: &mut Vec<usize>,
) -> Option<tt::TokenTree> {
    let res: tt::TokenTree = match template {
        crate::TokenTree::Subtree(subtree) => expand_subtree(subtree, bindings, nesting)?.into(),
        crate::TokenTree::Repeat(repeat) => {
            let mut token_trees = Vec::new();
            nesting.push(0);
            while let Some(t) = expand_subtree(&repeat.subtree, bindings, nesting) {
                let idx = nesting.pop().unwrap();
                nesting.push(idx + 1);
                token_trees.push(t.into())
            }
            nesting.pop().unwrap();
            tt::Subtree {
                token_trees,
                delimiter: tt::Delimiter::None,
            }
            .into()
        }
        crate::TokenTree::Leaf(leaf) => match leaf {
            crate::Leaf::Ident(ident) => tt::Leaf::from(tt::Ident {
                text: ident.text.clone(),
            })
            .into(),
            crate::Leaf::Punct(punct) => tt::Leaf::from(punct.clone()).into(),
            crate::Leaf::Var(v) => bindings.get(&v.text, nesting)?.clone(),
            crate::Leaf::Literal(l) => tt::Leaf::from(tt::Literal {
                text: l.text.clone(),
            })
            .into(),
        },
    };
    Some(res)
}
