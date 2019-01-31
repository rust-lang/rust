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
    inner: FxHashMap<SmolStr, tt::TokenTree>,
}

fn match_lhs(pattern: &mbe::Subtree, input: &tt::Subtree) -> Option<Bindings> {
    Some(Bindings::default())
}

fn expand_rhs(template: &mbe::Subtree, bindings: &Bindings) -> Option<tt::Subtree> {
    None
}
