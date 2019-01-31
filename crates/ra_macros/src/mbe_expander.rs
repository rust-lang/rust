use rustc_hash::FxHashMap;
use smol_str::SmolStr;

use crate::{mbe, tt};

pub fn exapnd(rules: &mbe::MacroRules, input: &tt::Subtree) -> Option<tt::Subtree> {
    rules.rules.iter().find_map(|it| expand_rule(it, input))
}

fn expand_rule(rule: &mbe::Rule, input: &tt::Subtree) -> Option<tt::Subtree> {
    let bindings = match_lhs(&rule.lhs, input)?;
    expand_rhs(&rule.rhs, &bindings)
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

fn match_lhs(pattern: &mbe::Subtree, input: &tt::Subtree) -> Option<Bindings> {
    let mut res = Bindings::default();
    for pat in pattern.token_trees.iter() {
        match pat {
            mbe::TokenTree::Leaf(leaf) => match leaf {
                mbe::Leaf::Var(mbe::Var { text, kind }) => {
                    let kind = kind.clone()?;
                    match kind.as_str() {
                        "ident" => (),
                        _ => return None,
                    }
                }
                _ => return None,
            },
            _ => {}
        }
    }
    Some(res)
}

fn expand_rhs(template: &mbe::Subtree, bindings: &Bindings) -> Option<tt::Subtree> {
    None
}
