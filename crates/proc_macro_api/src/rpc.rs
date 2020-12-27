//! Data structure serialization related stuff for RPC
//!
//! Defines all necessary rpc serialization data structures,
//! which includes `tt` related data and some task messages.
//! Although adding `Serialize` and `Deserialize` traits to `tt` directly seems
//! to be much easier, we deliberately duplicate `tt` structs with `#[serde(with = "XXDef")]`
//! for separation of code responsibility.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tt::{
    Delimiter, DelimiterKind, Ident, Leaf, Literal, Punct, SmolStr, Spacing, Subtree, TokenId,
    TokenTree,
};

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

#[derive(Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct ExpansionTask {
    /// Argument of macro call.
    ///
    /// In custom derive this will be a struct or enum; in attribute-like macro - underlying
    /// item; in function-like macro - the macro body.
    #[serde(with = "SubtreeDef")]
    pub macro_body: Subtree,

    /// Name of macro to expand.
    ///
    /// In custom derive this is the name of the derived trait (`Serialize`, `Getters`, etc.).
    /// In attribute-like and function-like macros - single name of macro itself (`show_streams`).
    pub macro_name: String,

    /// Possible attributes for the attribute-like macros.
    #[serde(with = "opt_subtree_def")]
    pub attributes: Option<Subtree>,

    pub lib: PathBuf,

    /// Environment variables to set during macro expansion.
    pub env: Vec<(String, String)>,
}

#[derive(Clone, Eq, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct ExpansionResult {
    #[serde(with = "SubtreeDef")]
    pub expansion: Subtree,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "DelimiterKind")]
enum DelimiterKindDef {
    Parenthesis,
    Brace,
    Bracket,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "TokenId")]
struct TokenIdDef(u32);

#[derive(Serialize, Deserialize)]
#[serde(remote = "Delimiter")]
struct DelimiterDef {
    #[serde(with = "TokenIdDef")]
    id: TokenId,
    #[serde(with = "DelimiterKindDef")]
    kind: DelimiterKind,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Subtree")]
struct SubtreeDef {
    #[serde(default, with = "opt_delimiter_def")]
    delimiter: Option<Delimiter>,
    #[serde(with = "vec_token_tree")]
    token_trees: Vec<TokenTree>,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "TokenTree")]
enum TokenTreeDef {
    #[serde(with = "LeafDef")]
    Leaf(Leaf),
    #[serde(with = "SubtreeDef")]
    Subtree(Subtree),
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Leaf")]
enum LeafDef {
    #[serde(with = "LiteralDef")]
    Literal(Literal),
    #[serde(with = "PunctDef")]
    Punct(Punct),
    #[serde(with = "IdentDef")]
    Ident(Ident),
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Literal")]
struct LiteralDef {
    text: SmolStr,
    #[serde(with = "TokenIdDef")]
    id: TokenId,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Punct")]
struct PunctDef {
    char: char,
    #[serde(with = "SpacingDef")]
    spacing: Spacing,
    #[serde(with = "TokenIdDef")]
    id: TokenId,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Spacing")]
enum SpacingDef {
    Alone,
    Joint,
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "Ident")]
struct IdentDef {
    text: SmolStr,
    #[serde(with = "TokenIdDef")]
    id: TokenId,
}

mod opt_delimiter_def {
    use super::{Delimiter, DelimiterDef};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(value: &Option<Delimiter>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Helper<'a>(#[serde(with = "DelimiterDef")] &'a Delimiter);
        value.as_ref().map(Helper).serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Option<Delimiter>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper(#[serde(with = "DelimiterDef")] Delimiter);
        let helper = Option::deserialize(deserializer)?;
        Ok(helper.map(|Helper(external)| external))
    }
}

mod opt_subtree_def {
    use super::{Subtree, SubtreeDef};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(value: &Option<Subtree>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Helper<'a>(#[serde(with = "SubtreeDef")] &'a Subtree);
        value.as_ref().map(Helper).serialize(serializer)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Option<Subtree>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper(#[serde(with = "SubtreeDef")] Subtree);
        let helper = Option::deserialize(deserializer)?;
        Ok(helper.map(|Helper(external)| external))
    }
}

mod vec_token_tree {
    use super::{TokenTree, TokenTreeDef};
    use serde::{ser::SerializeSeq, Deserialize, Deserializer, Serialize, Serializer};

    pub(super) fn serialize<S>(value: &Vec<TokenTree>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct Helper<'a>(#[serde(with = "TokenTreeDef")] &'a TokenTree);

        let items: Vec<_> = value.iter().map(Helper).collect();
        let mut seq = serializer.serialize_seq(Some(items.len()))?;
        for element in items {
            seq.serialize_element(&element)?;
        }
        seq.end()
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Vec<TokenTree>, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper(#[serde(with = "TokenTreeDef")] TokenTree);

        let helper = Vec::deserialize(deserializer)?;
        Ok(helper.into_iter().map(|Helper(external)| external).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_token_tree() -> Subtree {
        let mut subtree = Subtree::default();
        subtree
            .token_trees
            .push(TokenTree::Leaf(Ident { text: "struct".into(), id: TokenId(0) }.into()));
        subtree
            .token_trees
            .push(TokenTree::Leaf(Ident { text: "Foo".into(), id: TokenId(1) }.into()));
        subtree.token_trees.push(TokenTree::Subtree(
            Subtree {
                delimiter: Some(Delimiter { id: TokenId(2), kind: DelimiterKind::Brace }),
                token_trees: vec![],
            }
            .into(),
        ));
        subtree
    }

    #[test]
    fn test_proc_macro_rpc_works() {
        let tt = fixture_token_tree();
        let task = ExpansionTask {
            macro_body: tt.clone(),
            macro_name: Default::default(),
            attributes: None,
            lib: Default::default(),
            env: Default::default(),
        };

        let json = serde_json::to_string(&task).unwrap();
        let back: ExpansionTask = serde_json::from_str(&json).unwrap();

        assert_eq!(task.macro_body, back.macro_body);

        let result = ExpansionResult { expansion: tt.clone() };
        let json = serde_json::to_string(&result).unwrap();
        let back: ExpansionResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result, back);
    }
}
