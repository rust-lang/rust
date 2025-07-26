//! Defines messages for cross-process message passing based on `ndjson` wire protocol
pub(crate) mod flat;

use std::io::{self, BufRead, Write};

use paths::Utf8PathBuf;
use serde::de::DeserializeOwned;
use serde_derive::{Deserialize, Serialize};

use crate::ProcMacroKind;

pub use self::flat::{
    FlatTree, SpanDataIndexMap, deserialize_span_data_index_map, serialize_span_data_index_map,
};
pub use span::TokenId;

// The versions of the server protocol
pub const NO_VERSION_CHECK_VERSION: u32 = 0;
pub const VERSION_CHECK_VERSION: u32 = 1;
pub const ENCODE_CLOSE_SPAN_VERSION: u32 = 2;
pub const HAS_GLOBAL_SPANS: u32 = 3;
pub const RUST_ANALYZER_SPAN_SUPPORT: u32 = 4;
/// Whether literals encode their kind as an additional u32 field and idents their rawness as a u32 field.
pub const EXTENDED_LEAF_DATA: u32 = 5;
pub const HASHED_AST_ID: u32 = 6;

/// Current API version of the proc-macro protocol.
pub const CURRENT_API_VERSION: u32 = HASHED_AST_ID;

/// Represents requests sent from the client to the proc-macro-srv.
#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    /// Retrieves a list of macros from a given dynamic library.
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ListMacros { dylib_path: Utf8PathBuf },

    /// Expands a procedural macro.
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ExpandMacro(Box<ExpandMacro>),

    /// Performs an API version check between the client and the server.
    /// Since [`VERSION_CHECK_VERSION`]
    ApiVersionCheck {},

    /// Sets server-specific configurations.
    /// Since [`RUST_ANALYZER_SPAN_SUPPORT`]
    SetConfig(ServerConfig),
}

/// Defines the mode used for handling span data.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
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
    /// Returns a list of available macros in a dynamic library.
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ListMacros(Result<Vec<(String, ProcMacroKind)>, String>),

    /// Returns result of a macro expansion.
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ExpandMacro(Result<FlatTree, PanicMessage>),

    /// Returns the API version supported by the server.
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ApiVersionCheck(u32),

    /// Confirms the application of a configuration update.
    /// Since [`RUST_ANALYZER_SPAN_SUPPORT`]
    SetConfig(ServerConfig),

    /// Returns the result of a macro expansion, including extended span data.
    /// Since [`RUST_ANALYZER_SPAN_SUPPORT`]
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
    fn read<R: BufRead>(
        from_proto: ProtocolRead<R>,
        inp: &mut R,
        buf: &mut String,
    ) -> io::Result<Option<Self>> {
        Ok(match from_proto(inp, buf)? {
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
    fn write<W: Write>(self, to_proto: ProtocolWrite<W>, out: &mut W) -> io::Result<()> {
        let text = serde_json::to_string(&self)?;
        to_proto(out, &text)
    }
}

impl Message for Request {}
impl Message for Response {}

/// Type alias for a function that reads protocol messages from a buffered input stream.
#[allow(type_alias_bounds)]
type ProtocolRead<R: BufRead> =
    for<'i, 'buf> fn(inp: &'i mut R, buf: &'buf mut String) -> io::Result<Option<&'buf String>>;
/// Type alias for a function that writes protocol messages to an output stream.
#[allow(type_alias_bounds)]
type ProtocolWrite<W: Write> = for<'o, 'msg> fn(out: &'o mut W, msg: &'msg str) -> io::Result<()>;

#[cfg(test)]
mod tests {
    use intern::{Symbol, sym};
    use span::{
        Edition, ROOT_ERASED_FILE_AST_ID, Span, SpanAnchor, SyntaxContext, TextRange, TextSize,
    };
    use tt::{
        Delimiter, DelimiterKind, Ident, Leaf, Literal, Punct, Spacing, TopSubtree,
        TopSubtreeBuilder,
    };

    use super::*;

    fn fixture_token_tree() -> TopSubtree<Span> {
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
                range: TextRange::empty(TextSize::new(19)),
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
        builder.push(Leaf::Literal(Literal {
            symbol: Symbol::intern("Foo"),
            span: Span {
                range: TextRange::at(TextSize::new(10), TextSize::of("\"Foo\"")),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            kind: tt::LitKind::Str,
            suffix: None,
        }));
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
        builder.push(Leaf::Literal(Literal {
            symbol: sym::INTEGER_0,
            span: Span {
                range: TextRange::at(TextSize::new(15), TextSize::of("0u32")),
                anchor,
                ctx: SyntaxContext::root(Edition::CURRENT),
            },
            kind: tt::LitKind::Integer,
            suffix: Some(sym::u32),
        }));
        builder.close(Span {
            range: TextRange::at(TextSize::new(19), TextSize::of('}')),
            anchor,
            ctx: SyntaxContext::root(Edition::CURRENT),
        });

        builder.build()
    }

    #[test]
    fn test_proc_macro_rpc_works() {
        let tt = fixture_token_tree();
        for v in RUST_ANALYZER_SPAN_SUPPORT..=CURRENT_API_VERSION {
            let mut span_data_table = Default::default();
            let task = ExpandMacro {
                data: ExpandMacroData {
                    macro_body: FlatTree::new(tt.view(), v, &mut span_data_table),
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

            assert!(
                tt == back.data.macro_body.to_subtree_resolved(v, &span_data_table),
                "version: {v}"
            );
        }
    }
}
