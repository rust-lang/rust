pub mod assists;
mod extend_selection;
mod folding_ranges;
mod line_index;
mod line_index_utils;
mod structure;
#[cfg(test)]
mod test_utils;
mod typing;
mod diagnostics;

pub use self::{
    assists::LocalEdit,
    extend_selection::extend_selection,
    folding_ranges::{folding_ranges, Fold, FoldKind},
    line_index::{LineCol, LineIndex},
    line_index_utils::translate_offset_with_edit,
    structure::{file_structure, StructureNode},
    typing::{join_lines, on_enter, on_dot_typed, on_eq_typed},
    diagnostics::diagnostics
};
use ra_text_edit::TextEditBuilder;
use ra_syntax::{
    algo::find_leaf_at_offset,
    ast::{self, AstNode},
    SourceFileNode,
    SyntaxKind::{self, *},
    SyntaxNodeRef, TextRange, TextUnit, Direction,
};
use rustc_hash::FxHashSet;

#[derive(Debug)]
pub struct HighlightedRange {
    pub range: TextRange,
    pub tag: &'static str,
}

#[derive(Debug, Copy, Clone)]
pub enum Severity {
    Error,
    WeakWarning,
}

#[derive(Debug)]
pub struct Diagnostic {
    pub range: TextRange,
    pub msg: String,
    pub severity: Severity,
    pub fix: Option<LocalEdit>,
}

pub fn matching_brace(file: &SourceFileNode, offset: TextUnit) -> Option<TextUnit> {
    const BRACES: &[SyntaxKind] = &[
        L_CURLY, R_CURLY, L_BRACK, R_BRACK, L_PAREN, R_PAREN, L_ANGLE, R_ANGLE,
    ];
    let (brace_node, brace_idx) = find_leaf_at_offset(file.syntax(), offset)
        .filter_map(|node| {
            let idx = BRACES.iter().position(|&brace| brace == node.kind())?;
            Some((node, idx))
        })
        .next()?;
    let parent = brace_node.parent()?;
    let matching_kind = BRACES[brace_idx ^ 1];
    let matching_node = parent
        .children()
        .find(|node| node.kind() == matching_kind)?;
    Some(matching_node.range().start())
}

pub fn highlight(root: SyntaxNodeRef) -> Vec<HighlightedRange> {
    // Visited nodes to handle highlighting priorities
    let mut highlighted = FxHashSet::default();
    let mut res = Vec::new();
    for node in root.descendants() {
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
                if let Some(macro_call) = ast::MacroCall::cast(node) {
                    if let Some(path) = macro_call.path() {
                        if let Some(segment) = path.segment() {
                            if let Some(name_ref) = segment.name_ref() {
                                highlighted.insert(name_ref.syntax());
                                let range_start = name_ref.syntax().range().start();
                                let mut range_end = name_ref.syntax().range().end();
                                for sibling in path.syntax().siblings(Direction::Next) {
                                    match sibling.kind() {
                                        EXCL | IDENT => range_end = sibling.range().end(),
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
        res.push(HighlightedRange {
            range: node.range(),
            tag,
        })
    }
    res
}

pub fn syntax_tree(file: &SourceFileNode) -> String {
    ::ra_syntax::utils::dump_tree(file.syntax())
}

pub fn find_node_at_offset<'a, N: AstNode<'a>>(
    syntax: SyntaxNodeRef<'a>,
    offset: TextUnit,
) -> Option<N> {
    find_leaf_at_offset(syntax, offset).find_map(|leaf| leaf.ancestors().find_map(N::cast))
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{add_cursor, assert_eq_dbg, assert_eq_text, extract_offset};

    use super::*;

    #[test]
    fn test_highlighting() {
        let file = SourceFileNode::parse(
            r#"
// comment
fn main() {}
    println!("Hello, {}!", 92);
"#,
        );
        let hls = highlight(file.syntax());
        assert_eq_dbg(
            r#"[HighlightedRange { range: [1; 11), tag: "comment" },
                HighlightedRange { range: [12; 14), tag: "keyword" },
                HighlightedRange { range: [15; 19), tag: "function" },
                HighlightedRange { range: [29; 37), tag: "macro" },
                HighlightedRange { range: [38; 50), tag: "string" },
                HighlightedRange { range: [52; 54), tag: "literal" }]"#,
            &hls,
        );
    }

    #[test]
    fn test_matching_brace() {
        fn do_check(before: &str, after: &str) {
            let (pos, before) = extract_offset(before);
            let file = SourceFileNode::parse(&before);
            let new_pos = match matching_brace(&file, pos) {
                None => pos,
                Some(pos) => pos,
            };
            let actual = add_cursor(&before, new_pos);
            assert_eq_text!(after, &actual);
        }

        do_check("struct Foo { a: i32, }<|>", "struct Foo <|>{ a: i32, }");
    }

}
