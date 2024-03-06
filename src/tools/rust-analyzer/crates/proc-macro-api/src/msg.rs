//! Defines messages for cross-process message passing based on `ndjson` wire protocol
pub(crate) mod flat;

use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::ProcMacroKind;

pub use crate::msg::flat::{
    deserialize_span_data_index_map, serialize_span_data_index_map, FlatTree, SpanDataIndexMap,
    TokenId,
};

// The versions of the server protocol
pub const NO_VERSION_CHECK_VERSION: u32 = 0;
pub const VERSION_CHECK_VERSION: u32 = 1;
pub const ENCODE_CLOSE_SPAN_VERSION: u32 = 2;
pub const HAS_GLOBAL_SPANS: u32 = 3;
pub const RUST_ANALYZER_SPAN_SUPPORT: u32 = 4;

pub const CURRENT_API_VERSION: u32 = RUST_ANALYZER_SPAN_SUPPORT;

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ListMacros { dylib_path: PathBuf },
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ExpandMacro(Box<ExpandMacro>),
    /// Since [`VERSION_CHECK_VERSION`]
    ApiVersionCheck {},
    /// Since [`RUST_ANALYZER_SPAN_SUPPORT`]
    SetConfig(ServerConfig),
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub enum SpanMode {
    #[default]
    Id,
    RustAnalyzer,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ListMacros(Result<Vec<(String, ProcMacroKind)>, String>),
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ExpandMacro(Result<FlatTree, PanicMessage>),
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ApiVersionCheck(u32),
    /// Since [`RUST_ANALYZER_SPAN_SUPPORT`]
    SetConfig(ServerConfig),
    /// Since [`RUST_ANALYZER_SPAN_SUPPORT`]
    ExpandMacroExtended(Result<ExpandMacroExtended, PanicMessage>),
}

#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ServerConfig {
    pub span_mode: SpanMode,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacroExtended {
    pub tree: FlatTree,
    pub span_data_table: Vec<u32>,
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
    /// marker for serde skip stuff
    #[serde(skip_serializing_if = "ExpnGlobals::skip_serializing_if")]
    #[serde(default)]
    pub has_global_spans: ExpnGlobals,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub span_data_table: Vec<u32>,
}

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct ExpnGlobals {
    #[serde(skip_serializing)]
    #[serde(default)]
    pub serialize: bool,
    pub def_site: usize,
    pub call_site: usize,
    pub mixed_site: usize,
}

impl ExpnGlobals {
    fn skip_serializing_if(&self) -> bool {
        !self.serialize
    }
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
    use base_db::FileId;
    use la_arena::RawIdx;
    use span::{ErasedFileAstId, Span, SpanAnchor, SyntaxContextId};
    use text_size::{TextRange, TextSize};
    use tt::{Delimiter, DelimiterKind, Ident, Leaf, Literal, Punct, Spacing, Subtree, TokenTree};

    use super::*;

    fn fixture_token_tree() -> Subtree<Span> {
        let anchor = SpanAnchor {
            file_id: FileId::from_raw(0),
            ast_id: ErasedFileAstId::from_raw(RawIdx::from(0)),
        };

        let token_trees = Box::new([
            TokenTree::Leaf(
                Ident {
                    text: "struct".into(),
                    span: Span {
                        range: TextRange::at(TextSize::new(0), TextSize::of("struct")),
                        anchor,
                        ctx: SyntaxContextId::ROOT,
                    },
                }
                .into(),
            ),
            TokenTree::Leaf(
                Ident {
                    text: "Foo".into(),
                    span: Span {
                        range: TextRange::at(TextSize::new(5), TextSize::of("Foo")),
                        anchor,
                        ctx: SyntaxContextId::ROOT,
                    },
                }
                .into(),
            ),
            TokenTree::Leaf(Leaf::Literal(Literal {
                text: "Foo".into(),

                span: Span {
                    range: TextRange::at(TextSize::new(8), TextSize::of("Foo")),
                    anchor,
                    ctx: SyntaxContextId::ROOT,
                },
            })),
            TokenTree::Leaf(Leaf::Punct(Punct {
                char: '@',
                span: Span {
                    range: TextRange::at(TextSize::new(11), TextSize::of('@')),
                    anchor,
                    ctx: SyntaxContextId::ROOT,
                },
                spacing: Spacing::Joint,
            })),
            TokenTree::Subtree(Subtree {
                delimiter: Delimiter {
                    open: Span {
                        range: TextRange::at(TextSize::new(12), TextSize::of('{')),
                        anchor,
                        ctx: SyntaxContextId::ROOT,
                    },
                    close: Span {
                        range: TextRange::at(TextSize::new(13), TextSize::of('}')),
                        anchor,
                        ctx: SyntaxContextId::ROOT,
                    },
                    kind: DelimiterKind::Brace,
                },
                token_trees: Box::new([]),
            }),
        ]);

        Subtree {
            delimiter: Delimiter {
                open: Span {
                    range: TextRange::empty(TextSize::new(0)),
                    anchor,
                    ctx: SyntaxContextId::ROOT,
                },
                close: Span {
                    range: TextRange::empty(TextSize::new(13)),
                    anchor,
                    ctx: SyntaxContextId::ROOT,
                },
                kind: DelimiterKind::Invisible,
            },
            token_trees,
        }
    }

    #[test]
    fn test_proc_macro_rpc_works() {
        let tt = fixture_token_tree();
        let mut span_data_table = Default::default();
        let task = ExpandMacro {
            macro_body: FlatTree::new(&tt, CURRENT_API_VERSION, &mut span_data_table),
            macro_name: Default::default(),
            attributes: None,
            lib: std::env::current_dir().unwrap(),
            env: Default::default(),
            current_dir: Default::default(),
            has_global_spans: ExpnGlobals {
                serialize: true,
                def_site: 0,
                call_site: 0,
                mixed_site: 0,
            },
            span_data_table: Vec::new(),
        };

        let json = serde_json::to_string(&task).unwrap();
        // println!("{}", json);
        let back: ExpandMacro = serde_json::from_str(&json).unwrap();

        assert_eq!(tt, back.macro_body.to_subtree_resolved(CURRENT_API_VERSION, &span_data_table));
    }
}
