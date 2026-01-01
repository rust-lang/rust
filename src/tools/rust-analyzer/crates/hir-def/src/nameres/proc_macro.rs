//! Nameres-specific procedural macro data and helpers.

use hir_expand::name::{AsName, Name};
use intern::sym;
use itertools::Itertools;

use crate::{
    item_tree::Attrs,
    tt::{Leaf, TopSubtree, TtElement},
};

#[derive(Debug, PartialEq, Eq)]
pub struct ProcMacroDef {
    pub name: Name,
    pub kind: ProcMacroKind,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ProcMacroKind {
    Derive { helpers: Box<[Name]> },
    Bang,
    Attr,
}

impl ProcMacroKind {
    pub(super) fn to_basedb_kind(&self) -> hir_expand::proc_macro::ProcMacroKind {
        match self {
            ProcMacroKind::Derive { .. } => hir_expand::proc_macro::ProcMacroKind::CustomDerive,
            ProcMacroKind::Bang => hir_expand::proc_macro::ProcMacroKind::Bang,
            ProcMacroKind::Attr => hir_expand::proc_macro::ProcMacroKind::Attr,
        }
    }
}

impl Attrs<'_> {
    pub(crate) fn parse_proc_macro_decl(&self, func_name: &Name) -> Option<ProcMacroDef> {
        if self.is_proc_macro() {
            Some(ProcMacroDef { name: func_name.clone(), kind: ProcMacroKind::Bang })
        } else if self.is_proc_macro_attribute() {
            Some(ProcMacroDef { name: func_name.clone(), kind: ProcMacroKind::Attr })
        } else if self.by_key(sym::proc_macro_derive).exists() {
            let derive = self.parse_proc_macro_derive();
            Some(match derive {
                Some((name, helpers)) => {
                    ProcMacroDef { name, kind: ProcMacroKind::Derive { helpers } }
                }
                None => ProcMacroDef {
                    name: func_name.clone(),
                    kind: ProcMacroKind::Derive { helpers: Box::default() },
                },
            })
        } else {
            None
        }
    }

    pub(crate) fn parse_proc_macro_derive(&self) -> Option<(Name, Box<[Name]>)> {
        let derive = self.by_key(sym::proc_macro_derive).tt_values().next()?;
        parse_macro_name_and_helper_attrs(derive)
    }
}

// This fn is intended for `#[proc_macro_derive(..)]` and `#[rustc_builtin_macro(..)]`, which have
// the same structure.
pub(crate) fn parse_macro_name_and_helper_attrs(tt: &TopSubtree) -> Option<(Name, Box<[Name]>)> {
    if let Some([TtElement::Leaf(Leaf::Ident(trait_name))]) =
        tt.token_trees().iter().collect_array()
    {
        // `#[proc_macro_derive(Trait)]`
        // `#[rustc_builtin_macro(Trait)]`
        Some((trait_name.as_name(), Box::new([])))
    } else if let Some(
        [
            TtElement::Leaf(Leaf::Ident(trait_name)),
            TtElement::Leaf(Leaf::Punct(comma)),
            TtElement::Leaf(Leaf::Ident(attributes)),
            TtElement::Subtree(_, helpers),
        ],
    ) = tt.token_trees().iter().collect_array()
        && comma.char == ','
        && attributes.sym == sym::attributes
    {
        // `#[proc_macro_derive(Trait, attributes(helper1, helper2, ...))]`
        // `#[rustc_builtin_macro(Trait, attributes(helper1, helper2, ...))]`
        let helpers = helpers
            .filter_map(|tt| match tt {
                TtElement::Leaf(Leaf::Ident(helper)) => Some(helper.as_name()),
                _ => None,
            })
            .collect::<Box<[_]>>();

        Some((trait_name.as_name(), helpers))
    } else {
        None
    }
}
