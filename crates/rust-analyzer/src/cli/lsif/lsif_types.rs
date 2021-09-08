use lsp_types::FoldingRange;
use serde::{Deserialize, Serialize};

pub(crate) type RangeId = lsp_types::NumberOrString;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum LocationOrRangeId {
    Location(lsp_types::Location),
    RangeId(RangeId),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Entry {
    pub(crate) id: lsp_types::NumberOrString,
    #[serde(flatten)]
    pub(crate) data: Element,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "type")]
pub(crate) enum Element {
    Vertex(Vertex),
    Edge(Edge),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ToolInfo {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    args: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) enum Encoding {
    /// Currently only 'utf-16' is supported due to the limitations in LSP.
    #[serde(rename = "utf-16")]
    Utf16,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "label")]
pub(crate) enum Vertex {
    #[serde(rename_all = "camelCase")]
    MetaData {
        /// The version of the LSIF format using semver notation. See https://semver.org/. Please note
        /// the version numbers starting with 0 don't adhere to semver and adopters have to assume
        /// that each new version is breaking.
        version: String,

        /// The project root (in form of an URI) used to compute this dump.
        project_root: lsp_types::Url,

        /// The string encoding used to compute line and character values in
        /// positions and ranges.
        position_encoding: Encoding,

        /// Information about the tool that created the dump
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_info: Option<ToolInfo>,
    },
    /// https://github.com/Microsoft/language-server-protocol/blob/master/indexFormat/specification.md#the-project-vertex
    Project(Project),
    Document(Document),
    /// https://github.com/Microsoft/language-server-protocol/blob/master/indexFormat/specification.md#ranges
    Range(lsp_types::Range),
    /// https://github.com/Microsoft/language-server-protocol/blob/master/indexFormat/specification.md#result-set
    ResultSet(ResultSet),

    // FIXME: support all kind of results
    DefinitionResult {
        result: DefinitionResultType,
    },
    FoldingRangeResult {
        result: Vec<FoldingRange>,
    },
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "label")]
pub(crate) enum Edge {
    Contains(EdgeData),
    RefersTo(EdgeData),
    Item(Item),

    // Methods
    #[serde(rename = "textDocument/definition")]
    Definition(EdgeData),
    #[serde(rename = "textDocument/declaration")]
    Declaration(EdgeData),
    #[serde(rename = "textDocument/hover")]
    Hover(EdgeData),
    #[serde(rename = "textDocument/references")]
    References(EdgeData),
    #[serde(rename = "textDocument/implementation")]
    Implementation(EdgeData),
    #[serde(rename = "textDocument/typeDefinition")]
    TypeDefinition(EdgeData),
    #[serde(rename = "textDocument/foldingRange")]
    FoldingRange(EdgeData),
    #[serde(rename = "textDocument/documentLink")]
    DocumentLink(EdgeData),
    #[serde(rename = "textDocument/documentSymbol")]
    DocumentSymbol(EdgeData),
    #[serde(rename = "textDocument/diagnostic")]
    Diagnostic(EdgeData),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct EdgeData {
    pub(crate) in_v: lsp_types::NumberOrString,
    pub(crate) out_v: lsp_types::NumberOrString,
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum DefinitionResultType {
    Scalar(LocationOrRangeId),
    Array(LocationOrRangeId),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(tag = "property")]
pub(crate) enum Item {
    Definition(EdgeData),
    Reference(EdgeData),
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Document {
    pub(crate) uri: lsp_types::Url,
    pub(crate) language_id: Language,
}

/// https://github.com/Microsoft/language-server-protocol/blob/master/indexFormat/specification.md#result-set
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ResultSet {
    #[serde(skip_serializing_if = "Option::is_none")]
    key: Option<String>,
}

/// https://github.com/Microsoft/language-server-protocol/blob/master/indexFormat/specification.md#the-project-vertex
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct Project {
    project_file: lsp_types::Url,
    language_id: Language,
}

/// https://github.com/Microsoft/language-server-protocol/issues/213
/// For examples, see: https://code.visualstudio.com/docs/languages/identifiers.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Language {
    Rust,
    TypeScript,
    #[serde(other)]
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata() {
        let data = Entry {
            id: lsp_types::NumberOrString::Number(1),
            data: Element::Vertex(Vertex::MetaData {
                version: String::from("0.4.3"),
                project_root: lsp_types::Url::from_file_path("/hello/world").unwrap(),
                position_encoding: Encoding::Utf16,
                tool_info: Some(ToolInfo {
                    name: String::from("lsif-tsc"),
                    args: Some(vec![String::from("-p"), String::from(".")]),
                    version: Some(String::from("0.7.2")),
                }),
            }),
        };
        let text = r#"{"id":1,"type":"vertex","label":"metaData","version":"0.4.3","projectRoot":"file:///hello/world","positionEncoding":"utf-16","toolInfo":{"name":"lsif-tsc","args":["-p","."],"version":"0.7.2"}}"#
            .replace(' ', "");
        assert_eq!(serde_json::to_string(&data).unwrap(), text);
        assert_eq!(serde_json::from_str::<Entry>(&text).unwrap(), data);
    }

    #[test]
    fn document() {
        let data = Entry {
            id: lsp_types::NumberOrString::Number(1),
            data: Element::Vertex(Vertex::Document(Document {
                uri: lsp_types::Url::from_file_path("/Users/dirkb/sample.ts").unwrap(),
                language_id: Language::TypeScript,
            })),
        };

        let text = r#"{ "id": 1, "type": "vertex", "label": "document", "uri": "file:///Users/dirkb/sample.ts", "languageId": "typescript" }"#
            .replace(' ', "");

        assert_eq!(serde_json::to_string(&data).unwrap(), text);
        assert_eq!(serde_json::from_str::<Entry>(&text).unwrap(), data);
    }

    #[test]
    fn range() {
        let data = Entry {
            id: lsp_types::NumberOrString::Number(4),
            data: Element::Vertex(Vertex::Range(lsp_types::Range::new(
                lsp_types::Position::new(0, 9),
                lsp_types::Position::new(0, 12),
            ))),
        };

        let text = r#"{ "id": 4, "type": "vertex", "label": "range", "start": { "line": 0, "character": 9}, "end": { "line": 0, "character": 12 } }"#
            .replace(' ', "");

        assert_eq!(serde_json::to_string(&data).unwrap(), text);
        assert_eq!(serde_json::from_str::<Entry>(&text).unwrap(), data);
    }

    #[test]
    fn contains() {
        let data = Entry {
            id: lsp_types::NumberOrString::Number(5),
            data: Element::Edge(Edge::Contains(EdgeData {
                in_v: lsp_types::NumberOrString::Number(4),
                out_v: lsp_types::NumberOrString::Number(1),
            })),
        };

        let text = r#"{ "id": 5, "type": "edge", "label": "contains", "outV": 1, "inV": 4}"#
            .replace(' ', "");

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap(),
            serde_json::to_value(&data).unwrap()
        );
    }

    #[test]
    fn refers_to() {
        let data = Entry {
            id: lsp_types::NumberOrString::Number(5),
            data: Element::Edge(Edge::RefersTo(EdgeData {
                in_v: lsp_types::NumberOrString::Number(2),
                out_v: lsp_types::NumberOrString::Number(3),
            })),
        };

        let text = r#"{ "id": 5, "type": "edge", "label": "refersTo", "outV": 3, "inV": 2}"#
            .replace(' ', "");

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap(),
            serde_json::to_value(&data).unwrap()
        );
    }

    #[test]
    fn result_set() {
        let data = Entry {
            id: lsp_types::NumberOrString::Number(2),
            data: Element::Vertex(Vertex::ResultSet(ResultSet { key: None })),
        };

        let text = r#"{ "id": 2, "type": "vertex", "label": "resultSet" }"#.replace(' ', "");

        assert_eq!(serde_json::to_string(&data).unwrap(), text);
        assert_eq!(serde_json::from_str::<Entry>(&text).unwrap(), data);

        let data = Entry {
            id: lsp_types::NumberOrString::Number(4),
            data: Element::Vertex(Vertex::ResultSet(ResultSet {
                key: Some(String::from("hello")),
            })),
        };

        let text = r#"{ "id": 4, "type": "vertex", "label": "resultSet", "key": "hello" }"#
            .replace(' ', "");

        assert_eq!(serde_json::to_string(&data).unwrap(), text);
        assert_eq!(serde_json::from_str::<Entry>(&text).unwrap(), data);
    }

    #[test]
    fn definition() {
        let data = Entry {
            id: lsp_types::NumberOrString::Number(21),
            data: Element::Edge(Edge::Item(Item::Definition(EdgeData {
                in_v: lsp_types::NumberOrString::Number(18),
                out_v: lsp_types::NumberOrString::Number(16),
            }))),
        };

        let text = r#"{ "id": 21, "type": "edge", "label": "item", "property": "definition", "outV": 16, "inV": 18}"#
            .replace(' ', "");

        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&text).unwrap(),
            serde_json::to_value(&data).unwrap()
        );
    }

    mod methods {
        use super::*;

        #[test]
        fn references() {
            let data = Entry {
                id: lsp_types::NumberOrString::Number(17),
                data: Element::Edge(Edge::References(EdgeData {
                    in_v: lsp_types::NumberOrString::Number(16),
                    out_v: lsp_types::NumberOrString::Number(15),
                })),
            };

            let text = r#"{ "id": 17, "type": "edge", "label": "textDocument/references", "outV": 15, "inV": 16 }"#;

            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                serde_json::to_value(&data).unwrap()
            );
        }

        #[test]
        fn definition() {
            let data = Entry {
                id: lsp_types::NumberOrString::Number(13),
                data: Element::Vertex(Vertex::DefinitionResult {
                    result: DefinitionResultType::Scalar(LocationOrRangeId::RangeId(
                        lsp_types::NumberOrString::Number(7),
                    )),
                }),
            };

            let text =
                r#"{ "id": 13, "type": "vertex", "label": "definitionResult", "result": 7 }"#;

            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                serde_json::to_value(&data).unwrap()
            );
        }
    }
}
