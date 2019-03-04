use ra_db::SourceDatabase;
use crate::db::RootDatabase;
use ra_syntax::{
    SourceFile, SyntaxNode, TextRange, AstNode,
    algo::{self, visit::{visitor, Visitor}}, ast::{self, AstToken}
};

pub use ra_db::FileId;

pub(crate) fn syntax_tree(
    db: &RootDatabase,
    file_id: FileId,
    text_range: Option<TextRange>,
) -> String {
    if let Some(text_range) = text_range {
        let file = db.parse(file_id);
        let node = algo::find_covering_node(file.syntax(), text_range);

        if let Some(tree) = syntax_tree_for_string(node, text_range) {
            return tree;
        }

        node.debug_dump()
    } else {
        db.parse(file_id).syntax().debug_dump()
    }
}

/// Attempts parsing the selected contents of a string literal
/// as rust syntax and returns its syntax tree
fn syntax_tree_for_string(node: &SyntaxNode, text_range: TextRange) -> Option<String> {
    // When the range is inside a string
    // we'll attempt parsing it as rust syntax
    // to provide the syntax tree of the contents of the string
    visitor()
        .visit(|node: &ast::String| syntax_tree_for_token(node, text_range))
        .visit(|node: &ast::RawString| syntax_tree_for_token(node, text_range))
        .accept(node)?
}

fn syntax_tree_for_token<T: AstToken>(node: &T, text_range: TextRange) -> Option<String> {
    // Range of the full node
    let node_range = node.syntax().range();
    let text = node.text().to_string();

    // We start at some point inside the node
    // Either we have selected the whole string
    // or our selection is inside it
    let start = text_range.start() - node_range.start();

    // how many characters we have selected
    let len = text_range.len().to_usize();

    let node_len = node_range.len().to_usize();

    let start = start.to_usize();

    // We want to cap our length
    let len = len.min(node_len);

    // Ensure our slice is inside the actual string
    let end = if start + len < text.len() { start + len } else { text.len() - start };

    let text = &text[start..end];

    // Remove possible extra string quotes from the start
    // and the end of the string
    let text = text
        .trim_start_matches('r')
        .trim_start_matches('#')
        .trim_start_matches('"')
        .trim_end_matches('#')
        .trim_end_matches('"')
        .trim()
        // Remove custom markers
        .replace("<|>", "");

    let parsed = SourceFile::parse(&text);

    // If the "file" parsed without errors,
    // return its syntax
    if parsed.errors().is_empty() {
        return Some(parsed.syntax().debug_dump());
    }

    None
}
