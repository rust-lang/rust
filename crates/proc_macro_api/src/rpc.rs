//! Data structure serialization related stuff for RPC
//!
//! Defines all necessary rpc serialization data structures,
//! which includes `tt` related data and some task messages.
//! Although adding `Serialize` and `Deserialize` traits to `tt` directly seems
//! to be much easier, we deliberately duplicate `tt` structs with `#[serde(with = "XXDef")]`
//! for separation of code responsibility.
pub(crate) mod flat;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::rpc::flat::FlatTree;

#[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct ListMacrosTask {
    pub lib: PathBuf,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub enum ProcMacroKind {
    CustomDerive,
    FuncLike,
    Attr,
}

#[derive(Clone, Eq, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct ListMacrosResult {
    pub macros: Vec<(String, ProcMacroKind)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpansionTask {
    /// Argument of macro call.
    ///
    /// In custom derive this will be a struct or enum; in attribute-like macro - underlying
    /// item; in function-like macro - the macro body.
    pub macro_body: FlatTree,

    /// Name of macro to expand.
    ///
    /// In custom derive this is the name of the derived trait (`Serialize`, `Getters`, etc.).
    /// In attribute-like and function-like macros - single name of macro itself (`show_streams`).
    pub macro_name: String,

    /// Possible attributes for the attribute-like macros.
    pub attributes: Option<FlatTree>,

    pub lib: PathBuf,

    /// Environment variables to set during macro expansion.
    pub env: Vec<(String, String)>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpansionResult {
    pub expansion: FlatTree,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tt::*;

    fn fixture_token_tree() -> Subtree {
        let mut subtree = Subtree::default();
        subtree
            .token_trees
            .push(TokenTree::Leaf(Ident { text: "struct".into(), id: TokenId(0) }.into()));
        subtree
            .token_trees
            .push(TokenTree::Leaf(Ident { text: "Foo".into(), id: TokenId(1) }.into()));
        subtree.token_trees.push(TokenTree::Leaf(Leaf::Literal(Literal {
            text: "Foo".into(),
            id: TokenId::unspecified(),
        })));
        subtree.token_trees.push(TokenTree::Leaf(Leaf::Punct(Punct {
            char: '@',
            id: TokenId::unspecified(),
            spacing: Spacing::Joint,
        })));
        subtree.token_trees.push(TokenTree::Subtree(Subtree {
            delimiter: Some(Delimiter { id: TokenId(2), kind: DelimiterKind::Brace }),
            token_trees: vec![],
        }));
        subtree
    }

    #[test]
    fn test_proc_macro_rpc_works() {
        let tt = fixture_token_tree();
        let task = ExpansionTask {
            macro_body: FlatTree::new(&tt),
            macro_name: Default::default(),
            attributes: None,
            lib: std::env::current_dir().unwrap(),
            env: Default::default(),
        };

        let json = serde_json::to_string(&task).unwrap();
        // println!("{}", json);
        let back: ExpansionTask = serde_json::from_str(&json).unwrap();

        assert_eq!(tt, back.macro_body.to_subtree());
    }
}
