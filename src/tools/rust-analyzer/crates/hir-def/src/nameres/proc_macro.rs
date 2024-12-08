//! Nameres-specific procedural macro data and helpers.

use hir_expand::name::{AsName, Name};
use intern::sym;

use crate::attr::Attrs;
use crate::tt::{Leaf, TokenTree};

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

impl Attrs {
    #[rustfmt::skip]
    pub fn parse_proc_macro_decl(&self, func_name: &Name) -> Option<ProcMacroDef> {
        if self.is_proc_macro() {
            Some(ProcMacroDef { name: func_name.clone(), kind: ProcMacroKind::Bang })
        } else if self.is_proc_macro_attribute() {
            Some(ProcMacroDef { name: func_name.clone(), kind: ProcMacroKind::Attr })
        } else if self.by_key(&sym::proc_macro_derive).exists() {
            let derive = self.by_key(&sym::proc_macro_derive).tt_values().next()?;
            let def = parse_macro_name_and_helper_attrs(&derive.token_trees)
                .map(|(name, helpers)| ProcMacroDef { name, kind: ProcMacroKind::Derive { helpers } });

            if def.is_none() {
                tracing::trace!("malformed `#[proc_macro_derive]`: {}", derive);
            }

            def
        } else {
            None
        }
    }
}

// This fn is intended for `#[proc_macro_derive(..)]` and `#[rustc_builtin_macro(..)]`, which have
// the same structure.
#[rustfmt::skip]
pub(crate) fn parse_macro_name_and_helper_attrs(tt: &[TokenTree]) -> Option<(Name, Box<[Name]>)> {
    match tt {
        // `#[proc_macro_derive(Trait)]`
        // `#[rustc_builtin_macro(Trait)]`
        [TokenTree::Leaf(Leaf::Ident(trait_name))] => Some((trait_name.as_name(), Box::new([]))),

        // `#[proc_macro_derive(Trait, attributes(helper1, helper2, ...))]`
        // `#[rustc_builtin_macro(Trait, attributes(helper1, helper2, ...))]`
        [
            TokenTree::Leaf(Leaf::Ident(trait_name)),
            TokenTree::Leaf(Leaf::Punct(comma)),
            TokenTree::Leaf(Leaf::Ident(attributes)),
            TokenTree::Subtree(helpers)
        ] if comma.char == ',' && attributes.sym == sym::attributes =>
        {
            let helpers = helpers
                .token_trees
                .iter()
                .filter(
                    |tt| !matches!(tt, TokenTree::Leaf(Leaf::Punct(comma)) if comma.char == ','),
                )
                .map(|tt| match tt {
                    TokenTree::Leaf(Leaf::Ident(helper)) => Some(helper.as_name()),
                    _ => None,
                })
                .collect::<Option<Box<[_]>>>()?;

            Some((trait_name.as_name(), helpers))
        }

        _ => None,
    }
}
