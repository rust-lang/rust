//! Nameres-specific procedural macro data and helpers.

use hir_expand::name::{AsName, Name};
use intern::sym;

use crate::attr::Attrs;
use crate::tt::{Leaf, TokenTree, TopSubtree, TtElement};

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
    pub fn parse_proc_macro_decl(&self, func_name: &Name) -> Option<ProcMacroDef> {
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

    pub fn parse_proc_macro_derive(&self) -> Option<(Name, Box<[Name]>)> {
        let derive = self.by_key(sym::proc_macro_derive).tt_values().next()?;
        parse_macro_name_and_helper_attrs(derive)
    }

    pub fn parse_rustc_builtin_macro(&self) -> Option<(Name, Box<[Name]>)> {
        let derive = self.by_key(sym::rustc_builtin_macro).tt_values().next()?;
        parse_macro_name_and_helper_attrs(derive)
    }
}

// This fn is intended for `#[proc_macro_derive(..)]` and `#[rustc_builtin_macro(..)]`, which have
// the same structure.
#[rustfmt::skip]
pub(crate) fn parse_macro_name_and_helper_attrs(tt: &TopSubtree) -> Option<(Name, Box<[Name]>)> {
    match tt.token_trees().flat_tokens() {
        // `#[proc_macro_derive(Trait)]`
        // `#[rustc_builtin_macro(Trait)]`
        [TokenTree::Leaf(Leaf::Ident(trait_name))] => Some((trait_name.as_name(), Box::new([]))),

        // `#[proc_macro_derive(Trait, attributes(helper1, helper2, ...))]`
        // `#[rustc_builtin_macro(Trait, attributes(helper1, helper2, ...))]`
        [
            TokenTree::Leaf(Leaf::Ident(trait_name)),
            TokenTree::Leaf(Leaf::Punct(comma)),
            TokenTree::Leaf(Leaf::Ident(attributes)),
            TokenTree::Subtree(_),
            ..
        ] if comma.char == ',' && attributes.sym == sym::attributes =>
        {
            let helpers = tt::TokenTreesView::new(&tt.token_trees().flat_tokens()[3..]).try_into_subtree()?;
            let helpers = helpers
                .iter()
                .filter(
                    |tt| !matches!(tt, TtElement::Leaf(Leaf::Punct(comma)) if comma.char == ','),
                )
                .map(|tt| match tt {
                    TtElement::Leaf(Leaf::Ident(helper)) => Some(helper.as_name()),
                    _ => None,
                })
                .collect::<Option<Box<[_]>>>()?;

            Some((trait_name.as_name(), helpers))
        }

        _ => None,
    }
}
