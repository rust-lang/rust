//! Defines messages for cross-process message passing based on `ndjson` wire protocol
pub(crate) mod flat;

use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::ProcMacroKind;

pub use crate::msg::flat::FlatTree;

// The versions of the server protocol
pub const NO_VERSION_CHECK_VERSION: u32 = 0;
pub const VERSION_CHECK_VERSION: u32 = 1;
pub const ENCODE_CLOSE_SPAN_VERSION: u32 = 2;

pub const CURRENT_API_VERSION: u32 = ENCODE_CLOSE_SPAN_VERSION;

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    ListMacros { dylib_path: PathBuf },
    ExpandMacro(ExpandMacro),
    ApiVersionCheck {},
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    ListMacros(Result<Vec<(String, ProcMacroKind)>, String>),
    ExpandMacro(Result<FlatTree, PanicMessage>),
    ApiVersionCheck(u32),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PanicMessage(pub String);

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacro {
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

    pub current_dir: Option<String>,
}

pub trait Message: Serialize + DeserializeOwned {
    fn read(inp: &mut impl BufRead, buf: &mut String) -> io::Result<Option<Self>> {
        Ok(match read_json(inp, buf)? {
            None => None,
            Some(text) => {
                let mut deserializer = serde_json::Deserializer::from_str(text);
                // Note that some proc-macro generate very deep syntax tree
                // We have to disable the current limit of serde here
                deserializer.disable_recursion_limit();
                Some(Self::deserialize(&mut deserializer)?)
            }
        })
    }
    fn write(self, out: &mut impl Write) -> io::Result<()> {
        let text = serde_json::to_string(&self)?;
        write_json(out, &text)
    }
}

impl Message for Request {}
impl Message for Response {}

fn read_json<'a>(inp: &mut impl BufRead, buf: &'a mut String) -> io::Result<Option<&'a String>> {
    loop {
        buf.clear();

        inp.read_line(buf)?;
        buf.pop(); // Remove trailing '\n'

        if buf.is_empty() {
            return Ok(None);
        }

        // Some ill behaved macro try to use stdout for debugging
        // We ignore it here
        if !buf.starts_with('{') {
            tracing::error!("proc-macro tried to print : {}", buf);
            continue;
        }

        return Ok(Some(buf));
    }
}

fn write_json(out: &mut impl Write, msg: &str) -> io::Result<()> {
    tracing::debug!("> {}", msg);
    out.write_all(msg.as_bytes())?;
    out.write_all(b"\n")?;
    out.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tt::*;

    fn fixture_token_tree() -> Subtree {
        let mut subtree = Subtree { delimiter: Delimiter::unspecified(), token_trees: Vec::new() };
        subtree
            .token_trees
            .push(TokenTree::Leaf(Ident { text: "struct".into(), span: TokenId(0) }.into()));
        subtree
            .token_trees
            .push(TokenTree::Leaf(Ident { text: "Foo".into(), span: TokenId(1) }.into()));
        subtree.token_trees.push(TokenTree::Leaf(Leaf::Literal(Literal {
            text: "Foo".into(),
            span: TokenId::unspecified(),
        })));
        subtree.token_trees.push(TokenTree::Leaf(Leaf::Punct(Punct {
            char: '@',
            span: TokenId::unspecified(),
            spacing: Spacing::Joint,
        })));
        subtree.token_trees.push(TokenTree::Subtree(Subtree {
            delimiter: Delimiter {
                open: TokenId(2),
                close: TokenId::UNSPECIFIED,
                kind: DelimiterKind::Brace,
            },
            token_trees: vec![],
        }));
        subtree
    }

    #[test]
    fn test_proc_macro_rpc_works() {
        let tt = fixture_token_tree();
        let task = ExpandMacro {
            macro_body: FlatTree::new(&tt, CURRENT_API_VERSION),
            macro_name: Default::default(),
            attributes: None,
            lib: std::env::current_dir().unwrap(),
            env: Default::default(),
            current_dir: Default::default(),
        };

        let json = serde_json::to_string(&task).unwrap();
        // println!("{}", json);
        let back: ExpandMacro = serde_json::from_str(&json).unwrap();

        assert_eq!(tt, back.macro_body.to_subtree(CURRENT_API_VERSION));
    }
}
