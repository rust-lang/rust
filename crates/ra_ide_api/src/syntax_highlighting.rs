use rustc_hash::FxHashSet;

use ra_syntax::{ast, AstNode, TextRange, Direction, SyntaxKind::*, SyntaxElement, T};
use ra_db::SourceDatabase;

use crate::{FileId, db::RootDatabase};

#[derive(Debug)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub tag: &'static str,
}

pub(crate) fn highlight(db: &RootDatabase, file_id: FileId) -> Vec<HighlightedRange> {
    let source_file = db.parse(file_id);

    // Visited nodes to handle highlighting priorities
    let mut highlighted: FxHashSet<SyntaxElement> = FxHashSet::default();
    let mut res = Vec::new();
    for node in source_file.syntax().descendants_with_tokens() {
        if highlighted.contains(&node) {
            continue;
        }
        let tag = match node.kind() {
            COMMENT => "comment",
            STRING | RAW_STRING | RAW_BYTE_STRING | BYTE_STRING => "string",
            ATTR => "attribute",
            NAME_REF => "text",
            NAME => "function",
            INT_NUMBER | FLOAT_NUMBER | CHAR | BYTE => "literal",
            LIFETIME => "parameter",
            k if k.is_keyword() => "keyword",
            _ => {
                if let Some(macro_call) = node.as_node().and_then(ast::MacroCall::cast) {
                    if let Some(path) = macro_call.path() {
                        if let Some(segment) = path.segment() {
                            if let Some(name_ref) = segment.name_ref() {
                                highlighted.insert(name_ref.syntax().into());
                                let range_start = name_ref.syntax().range().start();
                                let mut range_end = name_ref.syntax().range().end();
                                for sibling in path.syntax().siblings_with_tokens(Direction::Next) {
                                    match sibling.kind() {
                                        T![!] | IDENT => range_end = sibling.range().end(),
                                        _ => (),
                                    }
                                }
                                res.push(HighlightedRange {
                                    range: TextRange::from_to(range_start, range_end),
                                    tag: "macro",
                                })
                            }
                        }
                    }
                }
                continue;
            }
        };
        res.push(HighlightedRange { range: node.range(), tag })
    }
    res
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot_matches;

    use crate::mock_analysis::single_file;

    #[test]
    fn test_highlighting() {
        let (analysis, file_id) = single_file(
            r#"
// comment
fn main() {}
    println!("Hello, {}!", 92);
"#,
        );
        let result = analysis.highlight(file_id);
        assert_debug_snapshot_matches!("highlighting", result);
    }
}
