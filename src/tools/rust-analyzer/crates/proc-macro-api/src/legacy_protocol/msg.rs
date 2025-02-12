//! Defines messages for cross-process message passing based on `ndjson` wire protocol
pub(crate) mod flat;

use std::io::{self, BufRead, Write};

use paths::Utf8PathBuf;
use serde::de::DeserializeOwned;
use serde_derive::{Deserialize, Serialize};

use crate::ProcMacroKind;

pub use self::flat::{
    deserialize_span_data_index_map, serialize_span_data_index_map, FlatTree, SpanDataIndexMap,
};
pub use span::TokenId;

// The versions of the server protocol
pub const NO_VERSION_CHECK_VERSION: u32 = 0;
pub const VERSION_CHECK_VERSION: u32 = 1;
pub const ENCODE_CLOSE_SPAN_VERSION: u32 = 2;
pub const HAS_GLOBAL_SPANS: u32 = 3;
pub const RUST_ANALYZER_SPAN_SUPPORT: u32 = 4;
/// Whether literals encode their kind as an additional u32 field and idents their rawness as a u32 field
pub const EXTENDED_LEAF_DATA: u32 = 5;

pub const CURRENT_API_VERSION: u32 = EXTENDED_LEAF_DATA;

#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    /// Since [`NO_VERSION_CHECK_VERSION`]
    ListMacros { dylib_path: Utf8PathBuf },
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
    pub lib: Utf8PathBuf,
    /// Environment variables to set during macro expansion.
    pub env: Vec<(String, String)>,
    pub current_dir: Option<String>,
    #[serde(flatten)]
    pub data: ExpandMacroData,
}

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

#[allow(type_alias_bounds)]
type ProtocolRead<R: BufRead> =
    for<'i, 'buf> fn(inp: &'i mut R, buf: &'buf mut String) -> io::Result<Option<&'buf String>>;
#[allow(type_alias_bounds)]
type ProtocolWrite<W: Write> = for<'o, 'msg> fn(out: &'o mut W, msg: &'msg str) -> io::Result<()>;

#[cfg(test)]
mod tests {
    use intern::{sym, Symbol};
    use span::{Edition, ErasedFileAstId, Span, SpanAnchor, SyntaxContextId, TextRange, TextSize};
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
            ast_id: ErasedFileAstId::from_raw(0),
        };

        let mut builder = TopSubtreeBuilder::new(Delimiter {
            open: Span {
                range: TextRange::empty(TextSize::new(0)),
                anchor,
                ctx: SyntaxContextId::root(Edition::CURRENT),
            },
            close: Span {
                range: TextRange::empty(TextSize::new(19)),
                anchor,
                ctx: SyntaxContextId::root(Edition::CURRENT),
            },
            kind: DelimiterKind::Invisible,
        });

        builder.push(
            Ident {
                sym: Symbol::intern("struct"),
                span: Span {
                    range: TextRange::at(TextSize::new(0), TextSize::of("struct")),
                    anchor,
                    ctx: SyntaxContextId::root(Edition::CURRENT),
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
                    ctx: SyntaxContextId::root(Edition::CURRENT),
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
                ctx: SyntaxContextId::root(Edition::CURRENT),
            },
            kind: tt::LitKind::Str,
            suffix: None,
        }));
        builder.push(Leaf::Punct(Punct {
            char: '@',
            span: Span {
                range: TextRange::at(TextSize::new(13), TextSize::of('@')),
                anchor,
                ctx: SyntaxContextId::root(Edition::CURRENT),
            },
            spacing: Spacing::Joint,
        }));
        builder.open(
            DelimiterKind::Brace,
            Span {
                range: TextRange::at(TextSize::new(14), TextSize::of('{')),
                anchor,
                ctx: SyntaxContextId::root(Edition::CURRENT),
            },
        );
        builder.push(Leaf::Literal(Literal {
            symbol: sym::INTEGER_0.clone(),
            span: Span {
                range: TextRange::at(TextSize::new(15), TextSize::of("0u32")),
                anchor,
                ctx: SyntaxContextId::root(Edition::CURRENT),
            },
            kind: tt::LitKind::Integer,
            suffix: Some(sym::u32.clone()),
        }));
        builder.close(Span {
            range: TextRange::at(TextSize::new(19), TextSize::of('}')),
            anchor,
            ctx: SyntaxContextId::root(Edition::CURRENT),
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
