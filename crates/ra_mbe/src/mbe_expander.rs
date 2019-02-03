/// This module takes a (parsed) defenition of `macro_rules` invocation, a
/// `tt::TokenTree` representing an argument of macro invocation, and produces a
/// `tt::TokenTree` for the result of the expansion.
use rustc_hash::FxHashMap;
use ra_syntax::SmolStr;

use crate::tt_cursor::TtCursor;

pub(crate) fn exapnd(rules: &crate::MacroRules, input: &tt::Subtree) -> Option<tt::Subtree> {
    rules.rules.iter().find_map(|it| expand_rule(it, input))
}

fn expand_rule(rule: &crate::Rule, input: &tt::Subtree) -> Option<tt::Subtree> {
    let mut input = TtCursor::new(input);
    let bindings = match_lhs(&rule.lhs, &mut input)?;
    if !input.is_eof() {
        return None;
    }
    expand_subtree(&rule.rhs, &bindings, &mut Vec::new())
}

/// The actual algorithm for expansion is not too hard, but is pretty tricky.
/// `Bindings` structure is the key to understanding what we are doing here.
///
/// On the high level, it stores mapping from meta variables to the bits of
/// syntax it should be substituted with. For example, if `$e:expr` is matched
/// with `1 + 1` by macro_rules, the `Binding` will store `$e -> 1 + 1`.
///
/// The tricky bit is dealing with repetitions (`$()*`). Consider this example:
///
/// ```ignore
/// macro_rules! foo {
///     ($($ i:ident $($ e:expr),*);*) => {
///         $(fn $ i() { $($ e);*; })*
///     }
/// }
/// foo! { foo 1,2,3; bar 4,5,6 }
/// ```
///
/// Here, the `$i` meta variable is matched first with `foo` and then with
/// `bar`, and `$e` is matched in turn with `1`, `2`, `3`, `4`, `5`, `6`.
///
/// To represent such "multi-mappings", we use a recursive structures: we map
/// variables not to values, but to *lists* of values or other lists (that is,
/// to the trees).
///
/// For the above example, the bindings would store
///
/// ```ignore
/// i -> [foo, bar]
/// e -> [[1, 2, 3], [4, 5, 6]]
/// ```
///
/// We construct `Bindings` in the `match_lhs`. The interesting case is
/// `TokenTree::Repeat`, where we use `push_nested` to create the desired
/// nesting structure.
///
/// The other side of the puzzle is `expand_subtree`, where we use the bindings
/// to substitute meta variables in the output template. When expanding, we
/// maintain a `nesteing` stack of indicies whihc tells us which occurence from
/// the `Bindings` we should take. We push to the stack when we enter a
/// repetition.
///
/// In other words, `Bindings` is a *multi* mapping from `SmolStr` to
/// `tt::TokenTree`, where the index to select a particular `TokenTree` among
/// many is not a plain `usize`, but an `&[usize]`.
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
