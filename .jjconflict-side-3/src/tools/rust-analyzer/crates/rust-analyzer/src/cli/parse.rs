//! Read Rust code on stdin, print syntax tree on stdout.
use ide::Edition;
use ide_db::line_index::LineIndex;
use serde::Serialize;
use syntax::{AstNode, NodeOrToken, SourceFile, SyntaxNode, SyntaxToken};

use crate::cli::{flags, read_stdin};

#[derive(Serialize)]
struct JsonNode {
    kind: String,
    #[serde(rename = "type")]
    node_type: &'static str,
    start: [u32; 3],
    end: [u32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    children: Option<Vec<JsonNode>>,
}

fn pos(line_index: &LineIndex, offset: syntax::TextSize) -> [u32; 3] {
    let offset_u32 = u32::from(offset);
    let line_col = line_index.line_col(offset);
    [offset_u32, line_col.line, line_col.col]
}

impl flags::Parse {
    pub fn run(self) -> anyhow::Result<()> {
        let _p = tracing::info_span!("flags::Parse::run").entered();
        let text = read_stdin()?;
        let line_index = LineIndex::new(&text);
        let file = SourceFile::parse(&text, Edition::CURRENT).tree();

        if !self.no_dump {
            if self.json {
                let json_tree = node_to_json(NodeOrToken::Node(file.syntax().clone()), &line_index);
                println!("{}", serde_json::to_string(&json_tree)?);
            } else {
                println!("{:#?}", file.syntax());
            }
        }

        std::mem::forget(file);
        Ok(())
    }
}

fn node_to_json(node: NodeOrToken<SyntaxNode, SyntaxToken>, line_index: &LineIndex) -> JsonNode {
    let range = node.text_range();
    let kind = format!("{:?}", node.kind());

    match node {
        NodeOrToken::Node(n) => {
            let children: Vec<_> =
                n.children_with_tokens().map(|it| node_to_json(it, line_index)).collect();
            JsonNode {
                kind,
                node_type: "Node",
                start: pos(line_index, range.start()),
                end: pos(line_index, range.end()),
                text: None,
                children: Some(children),
            }
        }
        NodeOrToken::Token(t) => JsonNode {
            kind,
            node_type: "Token",
            start: pos(line_index, range.start()),
            end: pos(line_index, range.end()),
            text: Some(t.text().to_owned()),
            children: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::flags;

    #[test]
    fn test_parse_json_output() {
        let text = "fn main() {}".to_owned();
        let flags = flags::Parse { json: true, no_dump: false };
        let line_index = LineIndex::new(&text);

        let file = SourceFile::parse(&text, Edition::CURRENT).tree();

        let output = if flags.json {
            let json_tree = node_to_json(NodeOrToken::Node(file.syntax().clone()), &line_index);
            serde_json::to_string(&json_tree).unwrap()
        } else {
            format!("{:#?}", file.syntax())
        };

        assert!(output.contains(r#""kind":"SOURCE_FILE""#));
        assert!(output.contains(r#""text":"main""#));
        assert!(output.contains(r#""start":[0,0,0]"#));
    }
}
