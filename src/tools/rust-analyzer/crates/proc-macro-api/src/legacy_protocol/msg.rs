//! Defines messages for cross-process message passing based on `ndjson` wire protocol
pub(crate) mod flat;
pub use self::flat::*;

use std::io::{self, BufRead, Write};

use paths::Utf8PathBuf;
use serde::de::DeserializeOwned;
use serde_derive::{Deserialize, Serialize};

use crate::{ProcMacroKind, transport::json};

/// Represents requests sent from the client to the proc-macro-srv.
#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    // IMPORTANT: Keep his first, otherwise postcard will break as its not a self describing format
    // As such, this is the only request that needs to be supported across all protocol versions
    // and by keeping it first, we ensure it always has the same discriminant encoding in postcard
    /// Performs an API version check between the client and the server.
    /// Since [`crate::version::VERSION_CHECK_VERSION`]
    ApiVersionCheck {},

    /// Retrieves a list of macros from a given dynamic library.
    /// Since [`crate::version::NO_VERSION_CHECK_VERSION`]
    ListMacros { dylib_path: Utf8PathBuf },

    /// Expands a procedural macro.
    /// Since [`crate::version::NO_VERSION_CHECK_VERSION`]
    ExpandMacro(Box<ExpandMacro>),

    /// Sets server-specific configurations.
    /// Since [`crate::version::RUST_ANALYZER_SPAN_SUPPORT`]
    SetConfig(ServerConfig),
}

/// Defines the mode used for handling span data.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpanMode {
    /// Default mode, where spans are identified by an ID.
    #[default]
    Id,

    /// Rust Analyzer-specific span handling mode.
    RustAnalyzer,
}

/// Represents responses sent from the proc-macro-srv to the client.
#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    // IMPORTANT: Keep his first, otherwise postcard will break as its not a self describing format
    // As such, this is the only request that needs to be supported across all protocol versions
    // and by keeping it first, we ensure it always has the same discriminant encoding in postcard
    /// Returns the API version supported by the server.
    /// Since [`crate::version::NO_VERSION_CHECK_VERSION`]
    ApiVersionCheck(u32),

    /// Returns a list of available macros in a dynamic library.
    /// Since [`crate::version::NO_VERSION_CHECK_VERSION`]
    ListMacros(Result<Vec<(String, ProcMacroKind)>, String>),

    /// Returns result of a macro expansion.
    /// Since [`crate::version::NO_VERSION_CHECK_VERSION`]
    ExpandMacro(Result<FlatTree, PanicMessage>),

    /// Confirms the application of a configuration update.
    /// Since [`crate::version::RUST_ANALYZER_SPAN_SUPPORT`]
    SetConfig(ServerConfig),

    /// Returns the result of a macro expansion, including extended span data.
    /// Since [`crate::version::RUST_ANALYZER_SPAN_SUPPORT`]
    ExpandMacroExtended(Result<ExpandMacroExtended, PanicMessage>),
}

/// Configuration settings for the proc-macro-srv.
#[derive(Debug, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ServerConfig {
    /// Defines how span data should be handled.
    pub span_mode: SpanMode,
}

/// Represents an extended macro expansion response, including span data mappings.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacroExtended {
    /// The expanded syntax tree.
    pub tree: FlatTree,
    /// Additional span data mappings.
    pub span_data_table: Vec<u32>,
}

/// Represents an error message when a macro expansion results in a panic.
#[derive(Debug, Serialize, Deserialize)]
pub struct PanicMessage(pub String);

/// Represents a macro expansion request sent from the client.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacro {
    /// The path to the dynamic library containing the macro.
    pub lib: Utf8PathBuf,
    /// Environment variables to set during macro expansion.
    pub env: Vec<(String, String)>,
    /// The current working directory for the macro expansion.
    pub current_dir: Option<String>,
    /// Macro expansion data, including the macro body, name and attributes.
    #[serde(flatten)]
    pub data: ExpandMacroData,
}

/// Represents the input data required for expanding a macro.
#[derive(Debug, Serialize, Deserialize)]
pub struct ExpandMacroData {
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
    /// marker for serde skip stuff
    #[serde(skip_serializing_if = "ExpnGlobals::skip_serializing_if")]
    #[serde(default)]
    pub has_global_spans: ExpnGlobals,
    /// Table of additional span data.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub span_data_table: Vec<u32>,
}

/// Represents global expansion settings, including span resolution.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct ExpnGlobals {
    /// Determines whether to serialize the expansion settings.
    #[serde(skip_serializing)]
    #[serde(default)]
    pub serialize: bool,
    /// Defines the `def_site` span location.
    pub def_site: usize,
    /// Defines the `call_site` span location.
    pub call_site: usize,
    /// Defines the `mixed_site` span location.
    pub mixed_site: usize,
}

impl ExpnGlobals {
    fn skip_serializing_if(&self) -> bool {
        !self.serialize
    }
}

pub trait Message: serde::Serialize + DeserializeOwned {
    type Buf;
    fn read(inp: &mut dyn BufRead, buf: &mut Self::Buf) -> io::Result<Option<Self>>;
    fn write(self, out: &mut dyn Write) -> io::Result<()>;
}

impl Message for Request {
    type Buf = String;

    fn read(inp: &mut dyn BufRead, buf: &mut Self::Buf) -> io::Result<Option<Self>> {
        Ok(match json::read(inp, buf)? {
            None => None,
            Some(buf) => Some(json::decode(buf)?),
        })
    }
    fn write(self, out: &mut dyn Write) -> io::Result<()> {
        let value = json::encode(&self)?;
        json::write(out, &value)
    }
}

impl Message for Response {
    type Buf = String;

    fn read(inp: &mut dyn BufRead, buf: &mut Self::Buf) -> io::Result<Option<Self>> {
        Ok(match json::read(inp, buf)? {
            None => None,
            Some(buf) => Some(json::decode(buf)?),
        })
    }
    fn write(self, out: &mut dyn Write) -> io::Result<()> {
        let value = json::encode(&self)?;
        json::write(out, &value)
    }
}

#[cfg(test)]
mod tests {
    use intern::Symbol;
    use span::{
        Edition, ROOT_ERASED_FILE_AST_ID, Span, SpanAnchor, SyntaxContext, TextRange, TextSize,
    };
    use tt::{
        Delimiter, DelimiterKind, Ident, Leaf, Literal, Punct, Spacing, TopSubtree,
        TopSubtreeBuilder,
    };

    use crate::version;

    use super::*;

    fn fixture_token_tree_top_many_none() -> TopSubtree {
        let anchor = SpanAnchor {
            file_id: span::EditionedFileId::new(
                span::FileId::from_raw(0xe4e4e),
                span::Edition::CURRENT,
            ),
            ast_id: ROOT_ERASED_FILE_AST_ID,
        };

        let mut builder = TopSubtreeBuilder::new(Delimiter {
            open: Span {
                range: TextRange::empty(TextSize::new(0)),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            close: Span {
                range: TextRange::empty(TextSize::new(0)),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            kind: DelimiterKind::Invisible,
        });

        builder.push(
            Ident {
                sym: Symbol::intern("struct"),
                span: Span {
                    range: TextRange::at(TextSize::new(0), TextSize::of("struct")),
                    anchor,
                    ctx: SyntaxContext::root(Edition::CURRENT),
                },
                is_raw: tt::IdentIsRaw::No,
            }
            .into(),
        );
        builder.push(
            Ident {
                sym: Symbol::intern("Foo"),
                span: Span {
                    range: TextRange::at(TextSize::new(5), TextSize::of("r#Foo")),
                    anchor,
                    ctx: SyntaxContext::root(Edition::CURRENT),
                },
                is_raw: tt::IdentIsRaw::Yes,
            }
            .into(),
        );
        builder.push(Leaf::Literal(Literal::new_no_suffix(
            "Foo",
            Span {
                range: TextRange::at(TextSize::new(10), TextSize::of("\"Foo\"")),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            tt::LitKind::Str,
        )));
        builder.push(Leaf::Punct(Punct {
            char: '@',
            span: Span {
                range: TextRange::at(TextSize::new(13), TextSize::of('@')),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            spacing: Spacing::Joint,
        }));
        builder.open(
            DelimiterKind::Brace,
            Span {
                range: TextRange::at(TextSize::new(14), TextSize::of('{')),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
        );
        builder.open(
            DelimiterKind::Bracket,
            Span {
                range: TextRange::at(TextSize::new(15), TextSize::of('[')),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
        );
        builder.push(Leaf::Literal(Literal::new(
            "0",
            Span {
                range: TextRange::at(TextSize::new(16), TextSize::of("0u32")),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            tt::LitKind::Integer,
            "u32",
        )));
        builder.close(Span {
            range: TextRange::at(TextSize::new(20), TextSize::of(']')),
            anchor,
            ctx: SyntaxContext::root(Edition::CURRENT),
        });

        builder.close(Span {
            range: TextRange::at(TextSize::new(21), TextSize::of('}')),
            anchor,
            ctx: SyntaxContext::root(Edition::CURRENT),
        });

        builder.build()
    }

    fn fixture_token_tree_top_empty_none() -> TopSubtree {
        let anchor = SpanAnchor {
            file_id: span::EditionedFileId::new(
                span::FileId::from_raw(0xe4e4e),
                span::Edition::CURRENT,
            ),
            ast_id: ROOT_ERASED_FILE_AST_ID,
        };

        let builder = TopSubtreeBuilder::new(Delimiter {
            open: Span {
                range: TextRange::empty(TextSize::new(0)),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            close: Span {
                range: TextRange::empty(TextSize::new(0)),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            kind: DelimiterKind::Invisible,
        });

        builder.build()
    }

    fn fixture_token_tree_top_empty_brace() -> TopSubtree {
        let anchor = SpanAnchor {
            file_id: span::EditionedFileId::new(
                span::FileId::from_raw(0xe4e4e),
                span::Edition::CURRENT,
            ),
            ast_id: ROOT_ERASED_FILE_AST_ID,
        };

        let builder = TopSubtreeBuilder::new(Delimiter {
            open: Span {
                range: TextRange::empty(TextSize::new(0)),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            close: Span {
                range: TextRange::empty(TextSize::new(0)),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            kind: DelimiterKind::Brace,
        });

        builder.build()
    }

    #[test]
    fn test_proc_macro_rpc_works() {
        for tt in [
            fixture_token_tree_top_many_none,
            fixture_token_tree_top_empty_none,
            fixture_token_tree_top_empty_brace,
        ] {
            for v in version::RUST_ANALYZER_SPAN_SUPPORT..=version::CURRENT_API_VERSION {
                let tt = tt();
                let mut span_data_table = Default::default();
                let task = ExpandMacro {
                    data: ExpandMacroData {
                        macro_body: FlatTree::from_subtree(tt.view(), v, &mut span_data_table),
                        macro_name: Default::default(),
                        attributes: None,
                        has_global_spans: ExpnGlobals {
                            serialize: true,
                            def_site: 0,
                            call_site: 0,
                            mixed_site: 0,
                        },
                        span_data_table: Vec::new(),
                    },
                    lib: Utf8PathBuf::from_path_buf(std::env::current_dir().unwrap()).unwrap(),
                    env: Default::default(),
                    current_dir: Default::default(),
                };

                let json = serde_json::to_string(&task).unwrap();
                // println!("{}", json);
                let back: ExpandMacro = serde_json::from_str(&json).unwrap();

                assert_eq!(
                    tt,
                    back.data.macro_body.to_subtree_resolved(v, &span_data_table),
                    "version: {v}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "sysroot-abi")]
    fn test_proc_macro_rpc_works_ts() {
        for tt in [
            fixture_token_tree_top_many_none,
            fixture_token_tree_top_empty_none,
            fixture_token_tree_top_empty_brace,
        ] {
            let tt = tt();
            for v in version::RUST_ANALYZER_SPAN_SUPPORT..=version::CURRENT_API_VERSION {
                let mut span_data_table = Default::default();
                let flat_tree = FlatTree::from_subtree(tt.view(), v, &mut span_data_table);
                assert_eq!(
                    tt,
                    flat_tree.clone().to_subtree_resolved(v, &span_data_table),
                    "version: {v}"
                );
                let ts = flat_tree.to_tokenstream_resolved(v, &span_data_table, |a, b| a.cover(b));
                let call_site = *span_data_table.first().unwrap();
                let mut span_data_table = Default::default();
                assert_eq!(
                    tt,
                    FlatTree::from_tokenstream(ts.clone(), v, call_site, &mut span_data_table)
                        .to_subtree_resolved(v, &span_data_table),
                    "version: {v}, ts:\n{ts:#?}"
                );
            }
        }
    }
}
