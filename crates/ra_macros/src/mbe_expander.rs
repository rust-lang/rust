use crate::{mbe, tt};

pub fn exapnd(rules: &mbe::MacroRules, input: tt::Subtree) -> Option<tt::Subtree> {
    Some(input)
}
