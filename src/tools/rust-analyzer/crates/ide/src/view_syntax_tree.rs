use hir::Semantics;
use ide_db::{FileId, RootDatabase};
use span::TextRange;
use stdx::format_to;
use syntax::{
    ast::{self, IsString},
    AstNode, AstToken, NodeOrToken, SourceFile, SyntaxNode, SyntaxToken, WalkEvent,
};

// Feature: Show Syntax Tree
//
// Shows a tree view with the syntax tree of the current file
//
// |===
// | Editor  | Panel Name
//
// | VS Code | **Rust Syntax Tree**
// |===
pub(crate) fn view_syntax_tree(db: &RootDatabase, file_id: FileId) -> String {
    let sema = Semantics::new(db);
    let parse = sema.parse_guess_edition(file_id);
    syntax_node_to_json(parse.syntax(), None)
}

fn syntax_node_to_json(node: &SyntaxNode, ctx: Option<InStringCtx>) -> String {
    let mut result = String::new();
    for event in node.preorder_with_tokens() {
        match event {
            WalkEvent::Enter(it) => {
                let kind = it.kind();
                let (text_range, inner_range_str) = match &ctx {
                    Some(ctx) => {
                        let inner_start: u32 = it.text_range().start().into();
                        let inner_end: u32 = it.text_range().end().into();

                        let mut true_start = inner_start + ctx.offset;
                        let mut true_end = inner_end + ctx.offset;
                        for pos in &ctx.marker_positions {
                            if *pos >= inner_end {
                                break;
                            }

                            // We conditionally add to true_start in case
                            // the marker is between the start and end.
                            true_start += 2 * (*pos < inner_start) as u32;
                            true_end += 2;
                        }

                        let true_range = TextRange::new(true_start.into(), true_end.into());

                        (
                            true_range,
                            format!(
                                r#","istart":{:?},"iend":{:?}"#,
                                it.text_range().start(),
                                it.text_range().end()
                            ),
                        )
                    }
                    None => (it.text_range(), "".to_owned()),
                };
                let start = text_range.start();
                let end = text_range.end();

                match it {
                    NodeOrToken::Node(_) => {
                        format_to!(
                            result,
                            r#"{{"type":"Node","kind":"{kind:?}","start":{start:?},"end":{end:?}{inner_range_str},"children":["#
                        );
                    }
                    NodeOrToken::Token(token) => {
                        let comma = if token.next_sibling_or_token().is_some() { "," } else { "" };
                        match parse_rust_string(token) {
                            Some(parsed) => {
                                format_to!(
                                    result,
                                    r#"{{"type":"Node","kind":"{kind:?}","start":{start:?},"end":{end:?}{inner_range_str},"children":[{parsed}]}}{comma}"#
                                );
                            }
                            None => format_to!(
                                result,
                                r#"{{"type":"Token","kind":"{kind:?}","start":{start:?},"end":{end:?}{inner_range_str}}}{comma}"#
                            ),
                        }
                    }
                }
            }
            WalkEvent::Leave(it) => match it {
                NodeOrToken::Node(node) => {
                    let comma = if node.next_sibling_or_token().is_some() { "," } else { "" };
                    format_to!(result, "]}}{comma}")
                }
                NodeOrToken::Token(_) => (),
            },
        }
    }

    result
}

fn parse_rust_string(token: SyntaxToken) -> Option<String> {
    let string_node = ast::String::cast(token)?;
    let text = string_node.value().ok()?;

    let mut trim_result = String::new();
    let mut marker_positions = Vec::new();
    let mut skipped = 0;
    let mut last_end = 0;
    for (start, part) in text.match_indices("$0") {
        marker_positions.push((start - skipped) as u32);
        trim_result.push_str(&text[last_end..start]);
        skipped += part.len();
        last_end = start + part.len();
    }
    trim_result.push_str(&text[last_end..text.len()]);

    let parsed = SourceFile::parse(&trim_result, span::Edition::CURRENT);

    if !parsed.errors().is_empty() {
        return None;
    }

    let node: &SyntaxNode = &parsed.syntax_node();

    if node.children().count() == 0 {
        // C'mon, you should have at least one node other than SOURCE_FILE
        return None;
    }

    Some(syntax_node_to_json(
        node,
        Some(InStringCtx {
            offset: string_node.text_range_between_quotes()?.start().into(),
            marker_positions,
        }),
    ))
}

struct InStringCtx {
    offset: u32,
    marker_positions: Vec<u32>,
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::fixture;

    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: expect_test::Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let syn = analysis.view_syntax_tree(file_id).unwrap();
        expect.assert_eq(&syn)
    }

    #[test]
    fn view_syntax_tree() {
        // Basic syntax
        check(
            r#"fn foo() {}"#,
            expect![[
                r#"{"type":"Node","kind":"SOURCE_FILE","start":0,"end":11,"children":[{"type":"Node","kind":"FN","start":0,"end":11,"children":[{"type":"Token","kind":"FN_KW","start":0,"end":2},{"type":"Token","kind":"WHITESPACE","start":2,"end":3},{"type":"Node","kind":"NAME","start":3,"end":6,"children":[{"type":"Token","kind":"IDENT","start":3,"end":6}]},{"type":"Node","kind":"PARAM_LIST","start":6,"end":8,"children":[{"type":"Token","kind":"L_PAREN","start":6,"end":7},{"type":"Token","kind":"R_PAREN","start":7,"end":8}]},{"type":"Token","kind":"WHITESPACE","start":8,"end":9},{"type":"Node","kind":"BLOCK_EXPR","start":9,"end":11,"children":[{"type":"Node","kind":"STMT_LIST","start":9,"end":11,"children":[{"type":"Token","kind":"L_CURLY","start":9,"end":10},{"type":"Token","kind":"R_CURLY","start":10,"end":11}]}]}]}]}"#
            ]],
        );

        check(
            r#"
fn test() {
    assert!("
    fn foo() {
    }
    ", "");
}"#,
            expect![[
                r#"{"type":"Node","kind":"SOURCE_FILE","start":0,"end":60,"children":[{"type":"Node","kind":"FN","start":0,"end":60,"children":[{"type":"Token","kind":"FN_KW","start":0,"end":2},{"type":"Token","kind":"WHITESPACE","start":2,"end":3},{"type":"Node","kind":"NAME","start":3,"end":7,"children":[{"type":"Token","kind":"IDENT","start":3,"end":7}]},{"type":"Node","kind":"PARAM_LIST","start":7,"end":9,"children":[{"type":"Token","kind":"L_PAREN","start":7,"end":8},{"type":"Token","kind":"R_PAREN","start":8,"end":9}]},{"type":"Token","kind":"WHITESPACE","start":9,"end":10},{"type":"Node","kind":"BLOCK_EXPR","start":10,"end":60,"children":[{"type":"Node","kind":"STMT_LIST","start":10,"end":60,"children":[{"type":"Token","kind":"L_CURLY","start":10,"end":11},{"type":"Token","kind":"WHITESPACE","start":11,"end":16},{"type":"Node","kind":"EXPR_STMT","start":16,"end":58,"children":[{"type":"Node","kind":"MACRO_EXPR","start":16,"end":57,"children":[{"type":"Node","kind":"MACRO_CALL","start":16,"end":57,"children":[{"type":"Node","kind":"PATH","start":16,"end":22,"children":[{"type":"Node","kind":"PATH_SEGMENT","start":16,"end":22,"children":[{"type":"Node","kind":"NAME_REF","start":16,"end":22,"children":[{"type":"Token","kind":"IDENT","start":16,"end":22}]}]}]},{"type":"Token","kind":"BANG","start":22,"end":23},{"type":"Node","kind":"TOKEN_TREE","start":23,"end":57,"children":[{"type":"Token","kind":"L_PAREN","start":23,"end":24},{"type":"Node","kind":"STRING","start":24,"end":52,"children":[{"type":"Node","kind":"SOURCE_FILE","start":25,"end":51,"istart":0,"iend":26,"children":[{"type":"Token","kind":"WHITESPACE","start":25,"end":30,"istart":0,"iend":5},{"type":"Node","kind":"FN","start":30,"end":46,"istart":5,"iend":21,"children":[{"type":"Token","kind":"FN_KW","start":30,"end":32,"istart":5,"iend":7},{"type":"Token","kind":"WHITESPACE","start":32,"end":33,"istart":7,"iend":8},{"type":"Node","kind":"NAME","start":33,"end":36,"istart":8,"iend":11,"children":[{"type":"Token","kind":"IDENT","start":33,"end":36,"istart":8,"iend":11}]},{"type":"Node","kind":"PARAM_LIST","start":36,"end":38,"istart":11,"iend":13,"children":[{"type":"Token","kind":"L_PAREN","start":36,"end":37,"istart":11,"iend":12},{"type":"Token","kind":"R_PAREN","start":37,"end":38,"istart":12,"iend":13}]},{"type":"Token","kind":"WHITESPACE","start":38,"end":39,"istart":13,"iend":14},{"type":"Node","kind":"BLOCK_EXPR","start":39,"end":46,"istart":14,"iend":21,"children":[{"type":"Node","kind":"STMT_LIST","start":39,"end":46,"istart":14,"iend":21,"children":[{"type":"Token","kind":"L_CURLY","start":39,"end":40,"istart":14,"iend":15},{"type":"Token","kind":"WHITESPACE","start":40,"end":45,"istart":15,"iend":20},{"type":"Token","kind":"R_CURLY","start":45,"end":46,"istart":20,"iend":21}]}]}]},{"type":"Token","kind":"WHITESPACE","start":46,"end":51,"istart":21,"iend":26}]}]},{"type":"Token","kind":"COMMA","start":52,"end":53},{"type":"Token","kind":"WHITESPACE","start":53,"end":54},{"type":"Token","kind":"STRING","start":54,"end":56},{"type":"Token","kind":"R_PAREN","start":56,"end":57}]}]}]},{"type":"Token","kind":"SEMICOLON","start":57,"end":58}]},{"type":"Token","kind":"WHITESPACE","start":58,"end":59},{"type":"Token","kind":"R_CURLY","start":59,"end":60}]}]}]}]}"#
            ]],
        )
    }

    #[test]
    fn view_syntax_tree_inside_string() {
        check(
            r#"fn test() {
    assert!("
$0fn foo() {
}$0
fn bar() {
}
    ", "");
}"#,
            expect![[
                r#"{"type":"Node","kind":"SOURCE_FILE","start":0,"end":65,"children":[{"type":"Node","kind":"FN","start":0,"end":65,"children":[{"type":"Token","kind":"FN_KW","start":0,"end":2},{"type":"Token","kind":"WHITESPACE","start":2,"end":3},{"type":"Node","kind":"NAME","start":3,"end":7,"children":[{"type":"Token","kind":"IDENT","start":3,"end":7}]},{"type":"Node","kind":"PARAM_LIST","start":7,"end":9,"children":[{"type":"Token","kind":"L_PAREN","start":7,"end":8},{"type":"Token","kind":"R_PAREN","start":8,"end":9}]},{"type":"Token","kind":"WHITESPACE","start":9,"end":10},{"type":"Node","kind":"BLOCK_EXPR","start":10,"end":65,"children":[{"type":"Node","kind":"STMT_LIST","start":10,"end":65,"children":[{"type":"Token","kind":"L_CURLY","start":10,"end":11},{"type":"Token","kind":"WHITESPACE","start":11,"end":16},{"type":"Node","kind":"EXPR_STMT","start":16,"end":63,"children":[{"type":"Node","kind":"MACRO_EXPR","start":16,"end":62,"children":[{"type":"Node","kind":"MACRO_CALL","start":16,"end":62,"children":[{"type":"Node","kind":"PATH","start":16,"end":22,"children":[{"type":"Node","kind":"PATH_SEGMENT","start":16,"end":22,"children":[{"type":"Node","kind":"NAME_REF","start":16,"end":22,"children":[{"type":"Token","kind":"IDENT","start":16,"end":22}]}]}]},{"type":"Token","kind":"BANG","start":22,"end":23},{"type":"Node","kind":"TOKEN_TREE","start":23,"end":62,"children":[{"type":"Token","kind":"L_PAREN","start":23,"end":24},{"type":"Node","kind":"STRING","start":24,"end":57,"children":[{"type":"Node","kind":"SOURCE_FILE","start":25,"end":56,"istart":0,"iend":31,"children":[{"type":"Token","kind":"WHITESPACE","start":25,"end":26,"istart":0,"iend":1},{"type":"Node","kind":"FN","start":26,"end":38,"istart":1,"iend":13,"children":[{"type":"Token","kind":"FN_KW","start":26,"end":28,"istart":1,"iend":3},{"type":"Token","kind":"WHITESPACE","start":28,"end":29,"istart":3,"iend":4},{"type":"Node","kind":"NAME","start":29,"end":32,"istart":4,"iend":7,"children":[{"type":"Token","kind":"IDENT","start":29,"end":32,"istart":4,"iend":7}]},{"type":"Node","kind":"PARAM_LIST","start":32,"end":34,"istart":7,"iend":9,"children":[{"type":"Token","kind":"L_PAREN","start":32,"end":33,"istart":7,"iend":8},{"type":"Token","kind":"R_PAREN","start":33,"end":34,"istart":8,"iend":9}]},{"type":"Token","kind":"WHITESPACE","start":34,"end":35,"istart":9,"iend":10},{"type":"Node","kind":"BLOCK_EXPR","start":35,"end":38,"istart":10,"iend":13,"children":[{"type":"Node","kind":"STMT_LIST","start":35,"end":38,"istart":10,"iend":13,"children":[{"type":"Token","kind":"L_CURLY","start":35,"end":36,"istart":10,"iend":11},{"type":"Token","kind":"WHITESPACE","start":36,"end":37,"istart":11,"iend":12},{"type":"Token","kind":"R_CURLY","start":37,"end":38,"istart":12,"iend":13}]}]}]},{"type":"Token","kind":"WHITESPACE","start":38,"end":39,"istart":13,"iend":14},{"type":"Node","kind":"FN","start":39,"end":51,"istart":14,"iend":26,"children":[{"type":"Token","kind":"FN_KW","start":39,"end":41,"istart":14,"iend":16},{"type":"Token","kind":"WHITESPACE","start":41,"end":42,"istart":16,"iend":17},{"type":"Node","kind":"NAME","start":42,"end":45,"istart":17,"iend":20,"children":[{"type":"Token","kind":"IDENT","start":42,"end":45,"istart":17,"iend":20}]},{"type":"Node","kind":"PARAM_LIST","start":45,"end":47,"istart":20,"iend":22,"children":[{"type":"Token","kind":"L_PAREN","start":45,"end":46,"istart":20,"iend":21},{"type":"Token","kind":"R_PAREN","start":46,"end":47,"istart":21,"iend":22}]},{"type":"Token","kind":"WHITESPACE","start":47,"end":48,"istart":22,"iend":23},{"type":"Node","kind":"BLOCK_EXPR","start":48,"end":51,"istart":23,"iend":26,"children":[{"type":"Node","kind":"STMT_LIST","start":48,"end":51,"istart":23,"iend":26,"children":[{"type":"Token","kind":"L_CURLY","start":48,"end":49,"istart":23,"iend":24},{"type":"Token","kind":"WHITESPACE","start":49,"end":50,"istart":24,"iend":25},{"type":"Token","kind":"R_CURLY","start":50,"end":51,"istart":25,"iend":26}]}]}]},{"type":"Token","kind":"WHITESPACE","start":51,"end":56,"istart":26,"iend":31}]}]},{"type":"Token","kind":"COMMA","start":57,"end":58},{"type":"Token","kind":"WHITESPACE","start":58,"end":59},{"type":"Token","kind":"STRING","start":59,"end":61},{"type":"Token","kind":"R_PAREN","start":61,"end":62}]}]}]},{"type":"Token","kind":"SEMICOLON","start":62,"end":63}]},{"type":"Token","kind":"WHITESPACE","start":63,"end":64},{"type":"Token","kind":"R_CURLY","start":64,"end":65}]}]}]}]}"#
            ]],
        );

        // With a raw string
        check(
            r###"fn test() {
    assert!(r#"
$0fn foo() {
}$0
fn bar() {
}
    "#, "");
}"###,
            expect![[
                r#"{"type":"Node","kind":"SOURCE_FILE","start":0,"end":68,"children":[{"type":"Node","kind":"FN","start":0,"end":68,"children":[{"type":"Token","kind":"FN_KW","start":0,"end":2},{"type":"Token","kind":"WHITESPACE","start":2,"end":3},{"type":"Node","kind":"NAME","start":3,"end":7,"children":[{"type":"Token","kind":"IDENT","start":3,"end":7}]},{"type":"Node","kind":"PARAM_LIST","start":7,"end":9,"children":[{"type":"Token","kind":"L_PAREN","start":7,"end":8},{"type":"Token","kind":"R_PAREN","start":8,"end":9}]},{"type":"Token","kind":"WHITESPACE","start":9,"end":10},{"type":"Node","kind":"BLOCK_EXPR","start":10,"end":68,"children":[{"type":"Node","kind":"STMT_LIST","start":10,"end":68,"children":[{"type":"Token","kind":"L_CURLY","start":10,"end":11},{"type":"Token","kind":"WHITESPACE","start":11,"end":16},{"type":"Node","kind":"EXPR_STMT","start":16,"end":66,"children":[{"type":"Node","kind":"MACRO_EXPR","start":16,"end":65,"children":[{"type":"Node","kind":"MACRO_CALL","start":16,"end":65,"children":[{"type":"Node","kind":"PATH","start":16,"end":22,"children":[{"type":"Node","kind":"PATH_SEGMENT","start":16,"end":22,"children":[{"type":"Node","kind":"NAME_REF","start":16,"end":22,"children":[{"type":"Token","kind":"IDENT","start":16,"end":22}]}]}]},{"type":"Token","kind":"BANG","start":22,"end":23},{"type":"Node","kind":"TOKEN_TREE","start":23,"end":65,"children":[{"type":"Token","kind":"L_PAREN","start":23,"end":24},{"type":"Node","kind":"STRING","start":24,"end":60,"children":[{"type":"Node","kind":"SOURCE_FILE","start":27,"end":58,"istart":0,"iend":31,"children":[{"type":"Token","kind":"WHITESPACE","start":27,"end":28,"istart":0,"iend":1},{"type":"Node","kind":"FN","start":28,"end":40,"istart":1,"iend":13,"children":[{"type":"Token","kind":"FN_KW","start":28,"end":30,"istart":1,"iend":3},{"type":"Token","kind":"WHITESPACE","start":30,"end":31,"istart":3,"iend":4},{"type":"Node","kind":"NAME","start":31,"end":34,"istart":4,"iend":7,"children":[{"type":"Token","kind":"IDENT","start":31,"end":34,"istart":4,"iend":7}]},{"type":"Node","kind":"PARAM_LIST","start":34,"end":36,"istart":7,"iend":9,"children":[{"type":"Token","kind":"L_PAREN","start":34,"end":35,"istart":7,"iend":8},{"type":"Token","kind":"R_PAREN","start":35,"end":36,"istart":8,"iend":9}]},{"type":"Token","kind":"WHITESPACE","start":36,"end":37,"istart":9,"iend":10},{"type":"Node","kind":"BLOCK_EXPR","start":37,"end":40,"istart":10,"iend":13,"children":[{"type":"Node","kind":"STMT_LIST","start":37,"end":40,"istart":10,"iend":13,"children":[{"type":"Token","kind":"L_CURLY","start":37,"end":38,"istart":10,"iend":11},{"type":"Token","kind":"WHITESPACE","start":38,"end":39,"istart":11,"iend":12},{"type":"Token","kind":"R_CURLY","start":39,"end":40,"istart":12,"iend":13}]}]}]},{"type":"Token","kind":"WHITESPACE","start":40,"end":41,"istart":13,"iend":14},{"type":"Node","kind":"FN","start":41,"end":53,"istart":14,"iend":26,"children":[{"type":"Token","kind":"FN_KW","start":41,"end":43,"istart":14,"iend":16},{"type":"Token","kind":"WHITESPACE","start":43,"end":44,"istart":16,"iend":17},{"type":"Node","kind":"NAME","start":44,"end":47,"istart":17,"iend":20,"children":[{"type":"Token","kind":"IDENT","start":44,"end":47,"istart":17,"iend":20}]},{"type":"Node","kind":"PARAM_LIST","start":47,"end":49,"istart":20,"iend":22,"children":[{"type":"Token","kind":"L_PAREN","start":47,"end":48,"istart":20,"iend":21},{"type":"Token","kind":"R_PAREN","start":48,"end":49,"istart":21,"iend":22}]},{"type":"Token","kind":"WHITESPACE","start":49,"end":50,"istart":22,"iend":23},{"type":"Node","kind":"BLOCK_EXPR","start":50,"end":53,"istart":23,"iend":26,"children":[{"type":"Node","kind":"STMT_LIST","start":50,"end":53,"istart":23,"iend":26,"children":[{"type":"Token","kind":"L_CURLY","start":50,"end":51,"istart":23,"iend":24},{"type":"Token","kind":"WHITESPACE","start":51,"end":52,"istart":24,"iend":25},{"type":"Token","kind":"R_CURLY","start":52,"end":53,"istart":25,"iend":26}]}]}]},{"type":"Token","kind":"WHITESPACE","start":53,"end":58,"istart":26,"iend":31}]}]},{"type":"Token","kind":"COMMA","start":60,"end":61},{"type":"Token","kind":"WHITESPACE","start":61,"end":62},{"type":"Token","kind":"STRING","start":62,"end":64},{"type":"Token","kind":"R_PAREN","start":64,"end":65}]}]}]},{"type":"Token","kind":"SEMICOLON","start":65,"end":66}]},{"type":"Token","kind":"WHITESPACE","start":66,"end":67},{"type":"Token","kind":"R_CURLY","start":67,"end":68}]}]}]}]}"#
            ]],
        );

        // With a raw string
        check(
            r###"fn test() {
    assert!(r$0#"
fn foo() {
}
fn bar() {
}"$0#, "");
}"###,
            expect![[
                r#"{"type":"Node","kind":"SOURCE_FILE","start":0,"end":63,"children":[{"type":"Node","kind":"FN","start":0,"end":63,"children":[{"type":"Token","kind":"FN_KW","start":0,"end":2},{"type":"Token","kind":"WHITESPACE","start":2,"end":3},{"type":"Node","kind":"NAME","start":3,"end":7,"children":[{"type":"Token","kind":"IDENT","start":3,"end":7}]},{"type":"Node","kind":"PARAM_LIST","start":7,"end":9,"children":[{"type":"Token","kind":"L_PAREN","start":7,"end":8},{"type":"Token","kind":"R_PAREN","start":8,"end":9}]},{"type":"Token","kind":"WHITESPACE","start":9,"end":10},{"type":"Node","kind":"BLOCK_EXPR","start":10,"end":63,"children":[{"type":"Node","kind":"STMT_LIST","start":10,"end":63,"children":[{"type":"Token","kind":"L_CURLY","start":10,"end":11},{"type":"Token","kind":"WHITESPACE","start":11,"end":16},{"type":"Node","kind":"EXPR_STMT","start":16,"end":61,"children":[{"type":"Node","kind":"MACRO_EXPR","start":16,"end":60,"children":[{"type":"Node","kind":"MACRO_CALL","start":16,"end":60,"children":[{"type":"Node","kind":"PATH","start":16,"end":22,"children":[{"type":"Node","kind":"PATH_SEGMENT","start":16,"end":22,"children":[{"type":"Node","kind":"NAME_REF","start":16,"end":22,"children":[{"type":"Token","kind":"IDENT","start":16,"end":22}]}]}]},{"type":"Token","kind":"BANG","start":22,"end":23},{"type":"Node","kind":"TOKEN_TREE","start":23,"end":60,"children":[{"type":"Token","kind":"L_PAREN","start":23,"end":24},{"type":"Node","kind":"STRING","start":24,"end":55,"children":[{"type":"Node","kind":"SOURCE_FILE","start":27,"end":53,"istart":0,"iend":26,"children":[{"type":"Token","kind":"WHITESPACE","start":27,"end":28,"istart":0,"iend":1},{"type":"Node","kind":"FN","start":28,"end":40,"istart":1,"iend":13,"children":[{"type":"Token","kind":"FN_KW","start":28,"end":30,"istart":1,"iend":3},{"type":"Token","kind":"WHITESPACE","start":30,"end":31,"istart":3,"iend":4},{"type":"Node","kind":"NAME","start":31,"end":34,"istart":4,"iend":7,"children":[{"type":"Token","kind":"IDENT","start":31,"end":34,"istart":4,"iend":7}]},{"type":"Node","kind":"PARAM_LIST","start":34,"end":36,"istart":7,"iend":9,"children":[{"type":"Token","kind":"L_PAREN","start":34,"end":35,"istart":7,"iend":8},{"type":"Token","kind":"R_PAREN","start":35,"end":36,"istart":8,"iend":9}]},{"type":"Token","kind":"WHITESPACE","start":36,"end":37,"istart":9,"iend":10},{"type":"Node","kind":"BLOCK_EXPR","start":37,"end":40,"istart":10,"iend":13,"children":[{"type":"Node","kind":"STMT_LIST","start":37,"end":40,"istart":10,"iend":13,"children":[{"type":"Token","kind":"L_CURLY","start":37,"end":38,"istart":10,"iend":11},{"type":"Token","kind":"WHITESPACE","start":38,"end":39,"istart":11,"iend":12},{"type":"Token","kind":"R_CURLY","start":39,"end":40,"istart":12,"iend":13}]}]}]},{"type":"Token","kind":"WHITESPACE","start":40,"end":41,"istart":13,"iend":14},{"type":"Node","kind":"FN","start":41,"end":53,"istart":14,"iend":26,"children":[{"type":"Token","kind":"FN_KW","start":41,"end":43,"istart":14,"iend":16},{"type":"Token","kind":"WHITESPACE","start":43,"end":44,"istart":16,"iend":17},{"type":"Node","kind":"NAME","start":44,"end":47,"istart":17,"iend":20,"children":[{"type":"Token","kind":"IDENT","start":44,"end":47,"istart":17,"iend":20}]},{"type":"Node","kind":"PARAM_LIST","start":47,"end":49,"istart":20,"iend":22,"children":[{"type":"Token","kind":"L_PAREN","start":47,"end":48,"istart":20,"iend":21},{"type":"Token","kind":"R_PAREN","start":48,"end":49,"istart":21,"iend":22}]},{"type":"Token","kind":"WHITESPACE","start":49,"end":50,"istart":22,"iend":23},{"type":"Node","kind":"BLOCK_EXPR","start":50,"end":53,"istart":23,"iend":26,"children":[{"type":"Node","kind":"STMT_LIST","start":50,"end":53,"istart":23,"iend":26,"children":[{"type":"Token","kind":"L_CURLY","start":50,"end":51,"istart":23,"iend":24},{"type":"Token","kind":"WHITESPACE","start":51,"end":52,"istart":24,"iend":25},{"type":"Token","kind":"R_CURLY","start":52,"end":53,"istart":25,"iend":26}]}]}]}]}]},{"type":"Token","kind":"COMMA","start":55,"end":56},{"type":"Token","kind":"WHITESPACE","start":56,"end":57},{"type":"Token","kind":"STRING","start":57,"end":59},{"type":"Token","kind":"R_PAREN","start":59,"end":60}]}]}]},{"type":"Token","kind":"SEMICOLON","start":60,"end":61}]},{"type":"Token","kind":"WHITESPACE","start":61,"end":62},{"type":"Token","kind":"R_CURLY","start":62,"end":63}]}]}]}]}"#
            ]],
        );
    }
}
