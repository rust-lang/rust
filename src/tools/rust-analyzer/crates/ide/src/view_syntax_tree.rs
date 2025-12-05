use hir::Semantics;
use ide_db::{
    FileId, LineIndexDatabase, RootDatabase,
    line_index::{LineCol, LineIndex},
};
use span::{TextRange, TextSize};
use stdx::format_to;
use syntax::{
    AstNode, AstToken, NodeOrToken, SourceFile, SyntaxNode, SyntaxToken, WalkEvent,
    ast::{self, IsString},
};
use triomphe::Arc;

// Feature: Show Syntax Tree
//
// Shows a tree view with the syntax tree of the current file
//
// | Editor  | Panel Name |
// |---------|-------------|
// | VS Code | **Rust Syntax Tree** |
pub(crate) fn view_syntax_tree(db: &RootDatabase, file_id: FileId) -> String {
    let sema = Semantics::new(db);
    let line_index = db.line_index(file_id);
    let parse = sema.parse_guess_edition(file_id);

    let ctx = SyntaxTreeCtx { line_index, in_string: None };

    syntax_node_to_json(parse.syntax(), &ctx)
}

fn syntax_node_to_json(node: &SyntaxNode, ctx: &SyntaxTreeCtx) -> String {
    let mut result = String::new();
    for event in node.preorder_with_tokens() {
        match event {
            WalkEvent::Enter(it) => {
                let kind = it.kind();
                let (text_range, inner_range_str) = match &ctx.in_string {
                    Some(in_string) => {
                        let start_pos = TextPosition::new(&ctx.line_index, it.text_range().start());
                        let end_pos = TextPosition::new(&ctx.line_index, it.text_range().end());

                        let inner_start: u32 = it.text_range().start().into();
                        let inner_end: u32 = it.text_range().start().into();

                        let mut true_start = inner_start + in_string.offset;
                        let mut true_end = inner_end + in_string.offset;
                        for pos in &in_string.marker_positions {
                            if *pos >= inner_end {
                                break;
                            }

                            // We conditionally add to true_start in case
                            // the marker is between the start and end.
                            true_start += 2 * (*pos < inner_start) as u32;
                            true_end += 2;
                        }

                        let true_range = TextRange::new(true_start.into(), true_end.into());

                        (true_range, format!(r#","istart":{start_pos},"iend":{end_pos}"#,))
                    }
                    None => (it.text_range(), "".to_owned()),
                };

                let start = TextPosition::new(&ctx.line_index, text_range.start());
                let end = TextPosition::new(&ctx.line_index, text_range.end());

                match it {
                    NodeOrToken::Node(_) => {
                        format_to!(
                            result,
                            r#"{{"type":"Node","kind":"{kind:?}","start":{start},"end":{end}{inner_range_str},"children":["#
                        );
                    }
                    NodeOrToken::Token(token) => {
                        let comma = if token.next_sibling_or_token().is_some() { "," } else { "" };
                        match parse_rust_string(token, ctx) {
                            Some(parsed) => {
                                format_to!(
                                    result,
                                    r#"{{"type":"Node","kind":"{kind:?}","start":{start},"end":{end}{inner_range_str},"children":[{parsed}]}}{comma}"#
                                );
                            }
                            None => format_to!(
                                result,
                                r#"{{"type":"Token","kind":"{kind:?}","start":{start},"end":{end}{inner_range_str}}}{comma}"#
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

struct TextPosition {
    offset: TextSize,
    line: u32,
    col: u32,
}

impl std::fmt::Display for TextPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?},{},{}]", self.offset, self.line, self.col)
    }
}

impl TextPosition {
    pub(crate) fn new(line_index: &LineIndex, offset: TextSize) -> Self {
        let LineCol { line, col } = line_index.line_col(offset);
        Self { offset, line, col }
    }
}

fn parse_rust_string(token: SyntaxToken, ctx: &SyntaxTreeCtx) -> Option<String> {
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

    let ctx = SyntaxTreeCtx {
        line_index: ctx.line_index.clone(),
        in_string: Some(InStringCtx {
            offset: string_node.text_range_between_quotes()?.start().into(),
            marker_positions,
        }),
    };

    Some(syntax_node_to_json(node, &ctx))
}

struct SyntaxTreeCtx {
    line_index: Arc<LineIndex>,
    in_string: Option<InStringCtx>,
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
                r#"{"type":"Node","kind":"SOURCE_FILE","start":[0,0,0],"end":[11,0,11],"children":[{"type":"Node","kind":"FN","start":[0,0,0],"end":[11,0,11],"children":[{"type":"Token","kind":"FN_KW","start":[0,0,0],"end":[2,0,2]},{"type":"Token","kind":"WHITESPACE","start":[2,0,2],"end":[3,0,3]},{"type":"Node","kind":"NAME","start":[3,0,3],"end":[6,0,6],"children":[{"type":"Token","kind":"IDENT","start":[3,0,3],"end":[6,0,6]}]},{"type":"Node","kind":"PARAM_LIST","start":[6,0,6],"end":[8,0,8],"children":[{"type":"Token","kind":"L_PAREN","start":[6,0,6],"end":[7,0,7]},{"type":"Token","kind":"R_PAREN","start":[7,0,7],"end":[8,0,8]}]},{"type":"Token","kind":"WHITESPACE","start":[8,0,8],"end":[9,0,9]},{"type":"Node","kind":"BLOCK_EXPR","start":[9,0,9],"end":[11,0,11],"children":[{"type":"Node","kind":"STMT_LIST","start":[9,0,9],"end":[11,0,11],"children":[{"type":"Token","kind":"L_CURLY","start":[9,0,9],"end":[10,0,10]},{"type":"Token","kind":"R_CURLY","start":[10,0,10],"end":[11,0,11]}]}]}]}]}"#
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
                r#"{"type":"Node","kind":"SOURCE_FILE","start":[0,0,0],"end":[60,5,1],"children":[{"type":"Node","kind":"FN","start":[0,0,0],"end":[60,5,1],"children":[{"type":"Token","kind":"FN_KW","start":[0,0,0],"end":[2,0,2]},{"type":"Token","kind":"WHITESPACE","start":[2,0,2],"end":[3,0,3]},{"type":"Node","kind":"NAME","start":[3,0,3],"end":[7,0,7],"children":[{"type":"Token","kind":"IDENT","start":[3,0,3],"end":[7,0,7]}]},{"type":"Node","kind":"PARAM_LIST","start":[7,0,7],"end":[9,0,9],"children":[{"type":"Token","kind":"L_PAREN","start":[7,0,7],"end":[8,0,8]},{"type":"Token","kind":"R_PAREN","start":[8,0,8],"end":[9,0,9]}]},{"type":"Token","kind":"WHITESPACE","start":[9,0,9],"end":[10,0,10]},{"type":"Node","kind":"BLOCK_EXPR","start":[10,0,10],"end":[60,5,1],"children":[{"type":"Node","kind":"STMT_LIST","start":[10,0,10],"end":[60,5,1],"children":[{"type":"Token","kind":"L_CURLY","start":[10,0,10],"end":[11,0,11]},{"type":"Token","kind":"WHITESPACE","start":[11,0,11],"end":[16,1,4]},{"type":"Node","kind":"EXPR_STMT","start":[16,1,4],"end":[58,4,11],"children":[{"type":"Node","kind":"MACRO_EXPR","start":[16,1,4],"end":[57,4,10],"children":[{"type":"Node","kind":"MACRO_CALL","start":[16,1,4],"end":[57,4,10],"children":[{"type":"Node","kind":"PATH","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Node","kind":"PATH_SEGMENT","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Node","kind":"NAME_REF","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Token","kind":"IDENT","start":[16,1,4],"end":[22,1,10]}]}]}]},{"type":"Token","kind":"BANG","start":[22,1,10],"end":[23,1,11]},{"type":"Node","kind":"TOKEN_TREE","start":[23,1,11],"end":[57,4,10],"children":[{"type":"Token","kind":"L_PAREN","start":[23,1,11],"end":[24,1,12]},{"type":"Node","kind":"STRING","start":[24,1,12],"end":[52,4,5],"children":[{"type":"Node","kind":"SOURCE_FILE","start":[25,1,13],"end":[25,1,13],"istart":[0,0,0],"iend":[26,2,0],"children":[{"type":"Token","kind":"WHITESPACE","start":[25,1,13],"end":[25,1,13],"istart":[0,0,0],"iend":[5,0,5]},{"type":"Node","kind":"FN","start":[30,2,4],"end":[30,2,4],"istart":[5,0,5],"iend":[21,1,9],"children":[{"type":"Token","kind":"FN_KW","start":[30,2,4],"end":[30,2,4],"istart":[5,0,5],"iend":[7,0,7]},{"type":"Token","kind":"WHITESPACE","start":[32,2,6],"end":[32,2,6],"istart":[7,0,7],"iend":[8,0,8]},{"type":"Node","kind":"NAME","start":[33,2,7],"end":[33,2,7],"istart":[8,0,8],"iend":[11,0,11],"children":[{"type":"Token","kind":"IDENT","start":[33,2,7],"end":[33,2,7],"istart":[8,0,8],"iend":[11,0,11]}]},{"type":"Node","kind":"PARAM_LIST","start":[36,2,10],"end":[36,2,10],"istart":[11,0,11],"iend":[13,1,1],"children":[{"type":"Token","kind":"L_PAREN","start":[36,2,10],"end":[36,2,10],"istart":[11,0,11],"iend":[12,1,0]},{"type":"Token","kind":"R_PAREN","start":[37,2,11],"end":[37,2,11],"istart":[12,1,0],"iend":[13,1,1]}]},{"type":"Token","kind":"WHITESPACE","start":[38,2,12],"end":[38,2,12],"istart":[13,1,1],"iend":[14,1,2]},{"type":"Node","kind":"BLOCK_EXPR","start":[39,2,13],"end":[39,2,13],"istart":[14,1,2],"iend":[21,1,9],"children":[{"type":"Node","kind":"STMT_LIST","start":[39,2,13],"end":[39,2,13],"istart":[14,1,2],"iend":[21,1,9],"children":[{"type":"Token","kind":"L_CURLY","start":[39,2,13],"end":[39,2,13],"istart":[14,1,2],"iend":[15,1,3]},{"type":"Token","kind":"WHITESPACE","start":[40,2,14],"end":[40,2,14],"istart":[15,1,3],"iend":[20,1,8]},{"type":"Token","kind":"R_CURLY","start":[45,3,4],"end":[45,3,4],"istart":[20,1,8],"iend":[21,1,9]}]}]}]},{"type":"Token","kind":"WHITESPACE","start":[46,3,5],"end":[46,3,5],"istart":[21,1,9],"iend":[26,2,0]}]}]},{"type":"Token","kind":"COMMA","start":[52,4,5],"end":[53,4,6]},{"type":"Token","kind":"WHITESPACE","start":[53,4,6],"end":[54,4,7]},{"type":"Token","kind":"STRING","start":[54,4,7],"end":[56,4,9]},{"type":"Token","kind":"R_PAREN","start":[56,4,9],"end":[57,4,10]}]}]}]},{"type":"Token","kind":"SEMICOLON","start":[57,4,10],"end":[58,4,11]}]},{"type":"Token","kind":"WHITESPACE","start":[58,4,11],"end":[59,5,0]},{"type":"Token","kind":"R_CURLY","start":[59,5,0],"end":[60,5,1]}]}]}]}]}"#
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
                r#"{"type":"Node","kind":"SOURCE_FILE","start":[0,0,0],"end":[65,7,1],"children":[{"type":"Node","kind":"FN","start":[0,0,0],"end":[65,7,1],"children":[{"type":"Token","kind":"FN_KW","start":[0,0,0],"end":[2,0,2]},{"type":"Token","kind":"WHITESPACE","start":[2,0,2],"end":[3,0,3]},{"type":"Node","kind":"NAME","start":[3,0,3],"end":[7,0,7],"children":[{"type":"Token","kind":"IDENT","start":[3,0,3],"end":[7,0,7]}]},{"type":"Node","kind":"PARAM_LIST","start":[7,0,7],"end":[9,0,9],"children":[{"type":"Token","kind":"L_PAREN","start":[7,0,7],"end":[8,0,8]},{"type":"Token","kind":"R_PAREN","start":[8,0,8],"end":[9,0,9]}]},{"type":"Token","kind":"WHITESPACE","start":[9,0,9],"end":[10,0,10]},{"type":"Node","kind":"BLOCK_EXPR","start":[10,0,10],"end":[65,7,1],"children":[{"type":"Node","kind":"STMT_LIST","start":[10,0,10],"end":[65,7,1],"children":[{"type":"Token","kind":"L_CURLY","start":[10,0,10],"end":[11,0,11]},{"type":"Token","kind":"WHITESPACE","start":[11,0,11],"end":[16,1,4]},{"type":"Node","kind":"EXPR_STMT","start":[16,1,4],"end":[63,6,11],"children":[{"type":"Node","kind":"MACRO_EXPR","start":[16,1,4],"end":[62,6,10],"children":[{"type":"Node","kind":"MACRO_CALL","start":[16,1,4],"end":[62,6,10],"children":[{"type":"Node","kind":"PATH","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Node","kind":"PATH_SEGMENT","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Node","kind":"NAME_REF","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Token","kind":"IDENT","start":[16,1,4],"end":[22,1,10]}]}]}]},{"type":"Token","kind":"BANG","start":[22,1,10],"end":[23,1,11]},{"type":"Node","kind":"TOKEN_TREE","start":[23,1,11],"end":[62,6,10],"children":[{"type":"Token","kind":"L_PAREN","start":[23,1,11],"end":[24,1,12]},{"type":"Node","kind":"STRING","start":[24,1,12],"end":[57,6,5],"children":[{"type":"Node","kind":"SOURCE_FILE","start":[25,1,13],"end":[25,1,13],"istart":[0,0,0],"iend":[31,2,5],"children":[{"type":"Token","kind":"WHITESPACE","start":[25,1,13],"end":[25,1,13],"istart":[0,0,0],"iend":[1,0,1]},{"type":"Node","kind":"FN","start":[26,2,0],"end":[26,2,0],"istart":[1,0,1],"iend":[13,1,1],"children":[{"type":"Token","kind":"FN_KW","start":[26,2,0],"end":[26,2,0],"istart":[1,0,1],"iend":[3,0,3]},{"type":"Token","kind":"WHITESPACE","start":[28,2,2],"end":[28,2,2],"istart":[3,0,3],"iend":[4,0,4]},{"type":"Node","kind":"NAME","start":[29,2,3],"end":[29,2,3],"istart":[4,0,4],"iend":[7,0,7],"children":[{"type":"Token","kind":"IDENT","start":[29,2,3],"end":[29,2,3],"istart":[4,0,4],"iend":[7,0,7]}]},{"type":"Node","kind":"PARAM_LIST","start":[32,2,6],"end":[32,2,6],"istart":[7,0,7],"iend":[9,0,9],"children":[{"type":"Token","kind":"L_PAREN","start":[32,2,6],"end":[32,2,6],"istart":[7,0,7],"iend":[8,0,8]},{"type":"Token","kind":"R_PAREN","start":[33,2,7],"end":[33,2,7],"istart":[8,0,8],"iend":[9,0,9]}]},{"type":"Token","kind":"WHITESPACE","start":[34,2,8],"end":[34,2,8],"istart":[9,0,9],"iend":[10,0,10]},{"type":"Node","kind":"BLOCK_EXPR","start":[35,2,9],"end":[35,2,9],"istart":[10,0,10],"iend":[13,1,1],"children":[{"type":"Node","kind":"STMT_LIST","start":[35,2,9],"end":[35,2,9],"istart":[10,0,10],"iend":[13,1,1],"children":[{"type":"Token","kind":"L_CURLY","start":[35,2,9],"end":[35,2,9],"istart":[10,0,10],"iend":[11,0,11]},{"type":"Token","kind":"WHITESPACE","start":[36,2,10],"end":[36,2,10],"istart":[11,0,11],"iend":[12,1,0]},{"type":"Token","kind":"R_CURLY","start":[37,3,0],"end":[37,3,0],"istart":[12,1,0],"iend":[13,1,1]}]}]}]},{"type":"Token","kind":"WHITESPACE","start":[38,3,1],"end":[38,3,1],"istart":[13,1,1],"iend":[14,1,2]},{"type":"Node","kind":"FN","start":[39,4,0],"end":[39,4,0],"istart":[14,1,2],"iend":[26,2,0],"children":[{"type":"Token","kind":"FN_KW","start":[39,4,0],"end":[39,4,0],"istart":[14,1,2],"iend":[16,1,4]},{"type":"Token","kind":"WHITESPACE","start":[41,4,2],"end":[41,4,2],"istart":[16,1,4],"iend":[17,1,5]},{"type":"Node","kind":"NAME","start":[42,4,3],"end":[42,4,3],"istart":[17,1,5],"iend":[20,1,8],"children":[{"type":"Token","kind":"IDENT","start":[42,4,3],"end":[42,4,3],"istart":[17,1,5],"iend":[20,1,8]}]},{"type":"Node","kind":"PARAM_LIST","start":[45,4,6],"end":[45,4,6],"istart":[20,1,8],"iend":[22,1,10],"children":[{"type":"Token","kind":"L_PAREN","start":[45,4,6],"end":[45,4,6],"istart":[20,1,8],"iend":[21,1,9]},{"type":"Token","kind":"R_PAREN","start":[46,4,7],"end":[46,4,7],"istart":[21,1,9],"iend":[22,1,10]}]},{"type":"Token","kind":"WHITESPACE","start":[47,4,8],"end":[47,4,8],"istart":[22,1,10],"iend":[23,1,11]},{"type":"Node","kind":"BLOCK_EXPR","start":[48,4,9],"end":[48,4,9],"istart":[23,1,11],"iend":[26,2,0],"children":[{"type":"Node","kind":"STMT_LIST","start":[48,4,9],"end":[48,4,9],"istart":[23,1,11],"iend":[26,2,0],"children":[{"type":"Token","kind":"L_CURLY","start":[48,4,9],"end":[48,4,9],"istart":[23,1,11],"iend":[24,1,12]},{"type":"Token","kind":"WHITESPACE","start":[49,4,10],"end":[49,4,10],"istart":[24,1,12],"iend":[25,1,13]},{"type":"Token","kind":"R_CURLY","start":[50,5,0],"end":[50,5,0],"istart":[25,1,13],"iend":[26,2,0]}]}]}]},{"type":"Token","kind":"WHITESPACE","start":[51,5,1],"end":[51,5,1],"istart":[26,2,0],"iend":[31,2,5]}]}]},{"type":"Token","kind":"COMMA","start":[57,6,5],"end":[58,6,6]},{"type":"Token","kind":"WHITESPACE","start":[58,6,6],"end":[59,6,7]},{"type":"Token","kind":"STRING","start":[59,6,7],"end":[61,6,9]},{"type":"Token","kind":"R_PAREN","start":[61,6,9],"end":[62,6,10]}]}]}]},{"type":"Token","kind":"SEMICOLON","start":[62,6,10],"end":[63,6,11]}]},{"type":"Token","kind":"WHITESPACE","start":[63,6,11],"end":[64,7,0]},{"type":"Token","kind":"R_CURLY","start":[64,7,0],"end":[65,7,1]}]}]}]}]}"#
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
                r#"{"type":"Node","kind":"SOURCE_FILE","start":[0,0,0],"end":[68,7,1],"children":[{"type":"Node","kind":"FN","start":[0,0,0],"end":[68,7,1],"children":[{"type":"Token","kind":"FN_KW","start":[0,0,0],"end":[2,0,2]},{"type":"Token","kind":"WHITESPACE","start":[2,0,2],"end":[3,0,3]},{"type":"Node","kind":"NAME","start":[3,0,3],"end":[7,0,7],"children":[{"type":"Token","kind":"IDENT","start":[3,0,3],"end":[7,0,7]}]},{"type":"Node","kind":"PARAM_LIST","start":[7,0,7],"end":[9,0,9],"children":[{"type":"Token","kind":"L_PAREN","start":[7,0,7],"end":[8,0,8]},{"type":"Token","kind":"R_PAREN","start":[8,0,8],"end":[9,0,9]}]},{"type":"Token","kind":"WHITESPACE","start":[9,0,9],"end":[10,0,10]},{"type":"Node","kind":"BLOCK_EXPR","start":[10,0,10],"end":[68,7,1],"children":[{"type":"Node","kind":"STMT_LIST","start":[10,0,10],"end":[68,7,1],"children":[{"type":"Token","kind":"L_CURLY","start":[10,0,10],"end":[11,0,11]},{"type":"Token","kind":"WHITESPACE","start":[11,0,11],"end":[16,1,4]},{"type":"Node","kind":"EXPR_STMT","start":[16,1,4],"end":[66,6,12],"children":[{"type":"Node","kind":"MACRO_EXPR","start":[16,1,4],"end":[65,6,11],"children":[{"type":"Node","kind":"MACRO_CALL","start":[16,1,4],"end":[65,6,11],"children":[{"type":"Node","kind":"PATH","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Node","kind":"PATH_SEGMENT","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Node","kind":"NAME_REF","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Token","kind":"IDENT","start":[16,1,4],"end":[22,1,10]}]}]}]},{"type":"Token","kind":"BANG","start":[22,1,10],"end":[23,1,11]},{"type":"Node","kind":"TOKEN_TREE","start":[23,1,11],"end":[65,6,11],"children":[{"type":"Token","kind":"L_PAREN","start":[23,1,11],"end":[24,1,12]},{"type":"Node","kind":"STRING","start":[24,1,12],"end":[60,6,6],"children":[{"type":"Node","kind":"SOURCE_FILE","start":[27,1,15],"end":[27,1,15],"istart":[0,0,0],"iend":[31,2,3],"children":[{"type":"Token","kind":"WHITESPACE","start":[27,1,15],"end":[27,1,15],"istart":[0,0,0],"iend":[1,0,1]},{"type":"Node","kind":"FN","start":[28,2,0],"end":[28,2,0],"istart":[1,0,1],"iend":[13,1,1],"children":[{"type":"Token","kind":"FN_KW","start":[28,2,0],"end":[28,2,0],"istart":[1,0,1],"iend":[3,0,3]},{"type":"Token","kind":"WHITESPACE","start":[30,2,2],"end":[30,2,2],"istart":[3,0,3],"iend":[4,0,4]},{"type":"Node","kind":"NAME","start":[31,2,3],"end":[31,2,3],"istart":[4,0,4],"iend":[7,0,7],"children":[{"type":"Token","kind":"IDENT","start":[31,2,3],"end":[31,2,3],"istart":[4,0,4],"iend":[7,0,7]}]},{"type":"Node","kind":"PARAM_LIST","start":[34,2,6],"end":[34,2,6],"istart":[7,0,7],"iend":[9,0,9],"children":[{"type":"Token","kind":"L_PAREN","start":[34,2,6],"end":[34,2,6],"istart":[7,0,7],"iend":[8,0,8]},{"type":"Token","kind":"R_PAREN","start":[35,2,7],"end":[35,2,7],"istart":[8,0,8],"iend":[9,0,9]}]},{"type":"Token","kind":"WHITESPACE","start":[36,2,8],"end":[36,2,8],"istart":[9,0,9],"iend":[10,0,10]},{"type":"Node","kind":"BLOCK_EXPR","start":[37,2,9],"end":[37,2,9],"istart":[10,0,10],"iend":[13,1,1],"children":[{"type":"Node","kind":"STMT_LIST","start":[37,2,9],"end":[37,2,9],"istart":[10,0,10],"iend":[13,1,1],"children":[{"type":"Token","kind":"L_CURLY","start":[37,2,9],"end":[37,2,9],"istart":[10,0,10],"iend":[11,0,11]},{"type":"Token","kind":"WHITESPACE","start":[38,2,10],"end":[38,2,10],"istart":[11,0,11],"iend":[12,1,0]},{"type":"Token","kind":"R_CURLY","start":[39,3,0],"end":[39,3,0],"istart":[12,1,0],"iend":[13,1,1]}]}]}]},{"type":"Token","kind":"WHITESPACE","start":[40,3,1],"end":[40,3,1],"istart":[13,1,1],"iend":[14,1,2]},{"type":"Node","kind":"FN","start":[41,4,0],"end":[41,4,0],"istart":[14,1,2],"iend":[26,1,14],"children":[{"type":"Token","kind":"FN_KW","start":[41,4,0],"end":[41,4,0],"istart":[14,1,2],"iend":[16,1,4]},{"type":"Token","kind":"WHITESPACE","start":[43,4,2],"end":[43,4,2],"istart":[16,1,4],"iend":[17,1,5]},{"type":"Node","kind":"NAME","start":[44,4,3],"end":[44,4,3],"istart":[17,1,5],"iend":[20,1,8],"children":[{"type":"Token","kind":"IDENT","start":[44,4,3],"end":[44,4,3],"istart":[17,1,5],"iend":[20,1,8]}]},{"type":"Node","kind":"PARAM_LIST","start":[47,4,6],"end":[47,4,6],"istart":[20,1,8],"iend":[22,1,10],"children":[{"type":"Token","kind":"L_PAREN","start":[47,4,6],"end":[47,4,6],"istart":[20,1,8],"iend":[21,1,9]},{"type":"Token","kind":"R_PAREN","start":[48,4,7],"end":[48,4,7],"istart":[21,1,9],"iend":[22,1,10]}]},{"type":"Token","kind":"WHITESPACE","start":[49,4,8],"end":[49,4,8],"istart":[22,1,10],"iend":[23,1,11]},{"type":"Node","kind":"BLOCK_EXPR","start":[50,4,9],"end":[50,4,9],"istart":[23,1,11],"iend":[26,1,14],"children":[{"type":"Node","kind":"STMT_LIST","start":[50,4,9],"end":[50,4,9],"istart":[23,1,11],"iend":[26,1,14],"children":[{"type":"Token","kind":"L_CURLY","start":[50,4,9],"end":[50,4,9],"istart":[23,1,11],"iend":[24,1,12]},{"type":"Token","kind":"WHITESPACE","start":[51,4,10],"end":[51,4,10],"istart":[24,1,12],"iend":[25,1,13]},{"type":"Token","kind":"R_CURLY","start":[52,5,0],"end":[52,5,0],"istart":[25,1,13],"iend":[26,1,14]}]}]}]},{"type":"Token","kind":"WHITESPACE","start":[53,5,1],"end":[53,5,1],"istart":[26,1,14],"iend":[31,2,3]}]}]},{"type":"Token","kind":"COMMA","start":[60,6,6],"end":[61,6,7]},{"type":"Token","kind":"WHITESPACE","start":[61,6,7],"end":[62,6,8]},{"type":"Token","kind":"STRING","start":[62,6,8],"end":[64,6,10]},{"type":"Token","kind":"R_PAREN","start":[64,6,10],"end":[65,6,11]}]}]}]},{"type":"Token","kind":"SEMICOLON","start":[65,6,11],"end":[66,6,12]}]},{"type":"Token","kind":"WHITESPACE","start":[66,6,12],"end":[67,7,0]},{"type":"Token","kind":"R_CURLY","start":[67,7,0],"end":[68,7,1]}]}]}]}]}"#
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
                r#"{"type":"Node","kind":"SOURCE_FILE","start":[0,0,0],"end":[63,6,1],"children":[{"type":"Node","kind":"FN","start":[0,0,0],"end":[63,6,1],"children":[{"type":"Token","kind":"FN_KW","start":[0,0,0],"end":[2,0,2]},{"type":"Token","kind":"WHITESPACE","start":[2,0,2],"end":[3,0,3]},{"type":"Node","kind":"NAME","start":[3,0,3],"end":[7,0,7],"children":[{"type":"Token","kind":"IDENT","start":[3,0,3],"end":[7,0,7]}]},{"type":"Node","kind":"PARAM_LIST","start":[7,0,7],"end":[9,0,9],"children":[{"type":"Token","kind":"L_PAREN","start":[7,0,7],"end":[8,0,8]},{"type":"Token","kind":"R_PAREN","start":[8,0,8],"end":[9,0,9]}]},{"type":"Token","kind":"WHITESPACE","start":[9,0,9],"end":[10,0,10]},{"type":"Node","kind":"BLOCK_EXPR","start":[10,0,10],"end":[63,6,1],"children":[{"type":"Node","kind":"STMT_LIST","start":[10,0,10],"end":[63,6,1],"children":[{"type":"Token","kind":"L_CURLY","start":[10,0,10],"end":[11,0,11]},{"type":"Token","kind":"WHITESPACE","start":[11,0,11],"end":[16,1,4]},{"type":"Node","kind":"EXPR_STMT","start":[16,1,4],"end":[61,5,9],"children":[{"type":"Node","kind":"MACRO_EXPR","start":[16,1,4],"end":[60,5,8],"children":[{"type":"Node","kind":"MACRO_CALL","start":[16,1,4],"end":[60,5,8],"children":[{"type":"Node","kind":"PATH","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Node","kind":"PATH_SEGMENT","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Node","kind":"NAME_REF","start":[16,1,4],"end":[22,1,10],"children":[{"type":"Token","kind":"IDENT","start":[16,1,4],"end":[22,1,10]}]}]}]},{"type":"Token","kind":"BANG","start":[22,1,10],"end":[23,1,11]},{"type":"Node","kind":"TOKEN_TREE","start":[23,1,11],"end":[60,5,8],"children":[{"type":"Token","kind":"L_PAREN","start":[23,1,11],"end":[24,1,12]},{"type":"Node","kind":"STRING","start":[24,1,12],"end":[55,5,3],"children":[{"type":"Node","kind":"SOURCE_FILE","start":[27,1,15],"end":[27,1,15],"istart":[0,0,0],"iend":[26,1,14],"children":[{"type":"Token","kind":"WHITESPACE","start":[27,1,15],"end":[27,1,15],"istart":[0,0,0],"iend":[1,0,1]},{"type":"Node","kind":"FN","start":[28,2,0],"end":[28,2,0],"istart":[1,0,1],"iend":[13,1,1],"children":[{"type":"Token","kind":"FN_KW","start":[28,2,0],"end":[28,2,0],"istart":[1,0,1],"iend":[3,0,3]},{"type":"Token","kind":"WHITESPACE","start":[30,2,2],"end":[30,2,2],"istart":[3,0,3],"iend":[4,0,4]},{"type":"Node","kind":"NAME","start":[31,2,3],"end":[31,2,3],"istart":[4,0,4],"iend":[7,0,7],"children":[{"type":"Token","kind":"IDENT","start":[31,2,3],"end":[31,2,3],"istart":[4,0,4],"iend":[7,0,7]}]},{"type":"Node","kind":"PARAM_LIST","start":[34,2,6],"end":[34,2,6],"istart":[7,0,7],"iend":[9,0,9],"children":[{"type":"Token","kind":"L_PAREN","start":[34,2,6],"end":[34,2,6],"istart":[7,0,7],"iend":[8,0,8]},{"type":"Token","kind":"R_PAREN","start":[35,2,7],"end":[35,2,7],"istart":[8,0,8],"iend":[9,0,9]}]},{"type":"Token","kind":"WHITESPACE","start":[36,2,8],"end":[36,2,8],"istart":[9,0,9],"iend":[10,0,10]},{"type":"Node","kind":"BLOCK_EXPR","start":[37,2,9],"end":[37,2,9],"istart":[10,0,10],"iend":[13,1,1],"children":[{"type":"Node","kind":"STMT_LIST","start":[37,2,9],"end":[37,2,9],"istart":[10,0,10],"iend":[13,1,1],"children":[{"type":"Token","kind":"L_CURLY","start":[37,2,9],"end":[37,2,9],"istart":[10,0,10],"iend":[11,0,11]},{"type":"Token","kind":"WHITESPACE","start":[38,2,10],"end":[38,2,10],"istart":[11,0,11],"iend":[12,1,0]},{"type":"Token","kind":"R_CURLY","start":[39,3,0],"end":[39,3,0],"istart":[12,1,0],"iend":[13,1,1]}]}]}]},{"type":"Token","kind":"WHITESPACE","start":[40,3,1],"end":[40,3,1],"istart":[13,1,1],"iend":[14,1,2]},{"type":"Node","kind":"FN","start":[41,4,0],"end":[41,4,0],"istart":[14,1,2],"iend":[26,1,14],"children":[{"type":"Token","kind":"FN_KW","start":[41,4,0],"end":[41,4,0],"istart":[14,1,2],"iend":[16,1,4]},{"type":"Token","kind":"WHITESPACE","start":[43,4,2],"end":[43,4,2],"istart":[16,1,4],"iend":[17,1,5]},{"type":"Node","kind":"NAME","start":[44,4,3],"end":[44,4,3],"istart":[17,1,5],"iend":[20,1,8],"children":[{"type":"Token","kind":"IDENT","start":[44,4,3],"end":[44,4,3],"istart":[17,1,5],"iend":[20,1,8]}]},{"type":"Node","kind":"PARAM_LIST","start":[47,4,6],"end":[47,4,6],"istart":[20,1,8],"iend":[22,1,10],"children":[{"type":"Token","kind":"L_PAREN","start":[47,4,6],"end":[47,4,6],"istart":[20,1,8],"iend":[21,1,9]},{"type":"Token","kind":"R_PAREN","start":[48,4,7],"end":[48,4,7],"istart":[21,1,9],"iend":[22,1,10]}]},{"type":"Token","kind":"WHITESPACE","start":[49,4,8],"end":[49,4,8],"istart":[22,1,10],"iend":[23,1,11]},{"type":"Node","kind":"BLOCK_EXPR","start":[50,4,9],"end":[50,4,9],"istart":[23,1,11],"iend":[26,1,14],"children":[{"type":"Node","kind":"STMT_LIST","start":[50,4,9],"end":[50,4,9],"istart":[23,1,11],"iend":[26,1,14],"children":[{"type":"Token","kind":"L_CURLY","start":[50,4,9],"end":[50,4,9],"istart":[23,1,11],"iend":[24,1,12]},{"type":"Token","kind":"WHITESPACE","start":[51,4,10],"end":[51,4,10],"istart":[24,1,12],"iend":[25,1,13]},{"type":"Token","kind":"R_CURLY","start":[52,5,0],"end":[52,5,0],"istart":[25,1,13],"iend":[26,1,14]}]}]}]}]}]},{"type":"Token","kind":"COMMA","start":[55,5,3],"end":[56,5,4]},{"type":"Token","kind":"WHITESPACE","start":[56,5,4],"end":[57,5,5]},{"type":"Token","kind":"STRING","start":[57,5,5],"end":[59,5,7]},{"type":"Token","kind":"R_PAREN","start":[59,5,7],"end":[60,5,8]}]}]}]},{"type":"Token","kind":"SEMICOLON","start":[60,5,8],"end":[61,5,9]}]},{"type":"Token","kind":"WHITESPACE","start":[61,5,9],"end":[62,6,0]},{"type":"Token","kind":"R_CURLY","start":[62,6,0],"end":[63,6,1]}]}]}]}]}"#
            ]],
        );
    }
}
