use ide_db::{
    base_db::{FileId, SourceDatabase},
    RootDatabase,
};
use syntax::{
    AstNode, NodeOrToken, SourceFile, SyntaxKind::STRING, SyntaxToken, TextRange, TextSize,
};

// Feature: Show Syntax Tree
//
// Shows the parse tree of the current file. It exists mostly for debugging
// rust-analyzer itself.
//
// |===
// | Editor  | Action Name
//
// | VS Code | **rust-analyzer: Show Syntax Tree**
// |===
// image::https://user-images.githubusercontent.com/48062697/113065586-068bdb80-91b1-11eb-9507-fee67f9f45a0.gif[]
pub(crate) fn syntax_tree(
    db: &RootDatabase,
    file_id: FileId,
    text_range: Option<TextRange>,
) -> String {
    let parse = db.parse(file_id);
    if let Some(text_range) = text_range {
        let node = match parse.tree().syntax().covering_element(text_range) {
            NodeOrToken::Node(node) => node,
            NodeOrToken::Token(token) => {
                if let Some(tree) = syntax_tree_for_string(&token, text_range) {
                    return tree;
                }
                token.parent().unwrap()
            }
        };

        format!("{node:#?}")
    } else {
        format!("{:#?}", parse.tree().syntax())
    }
}

/// Attempts parsing the selected contents of a string literal
/// as rust syntax and returns its syntax tree
fn syntax_tree_for_string(token: &SyntaxToken, text_range: TextRange) -> Option<String> {
    // When the range is inside a string
    // we'll attempt parsing it as rust syntax
    // to provide the syntax tree of the contents of the string
    match token.kind() {
        STRING => syntax_tree_for_token(token, text_range),
        _ => None,
    }
}

fn syntax_tree_for_token(node: &SyntaxToken, text_range: TextRange) -> Option<String> {
    // Range of the full node
    let node_range = node.text_range();
    let text = node.text().to_string();

    // We start at some point inside the node
    // Either we have selected the whole string
    // or our selection is inside it
    let start = text_range.start() - node_range.start();

    // how many characters we have selected
    let len = text_range.len();

    let node_len = node_range.len();

    let start = start;

    // We want to cap our length
    let len = len.min(node_len);

    // Ensure our slice is inside the actual string
    let end =
        if start + len < TextSize::of(&text) { start + len } else { TextSize::of(&text) - start };

    let text = &text[TextRange::new(start, end)];

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
        .replace("$0", "");

    let parsed = SourceFile::parse(&text);

    // If the "file" parsed without errors,
    // return its syntax
    if parsed.errors().is_empty() {
        return Some(format!("{:#?}", parsed.tree().syntax()));
    }

    None
}

#[cfg(test)]
mod tests {
    use expect_test::expect;

    use crate::fixture;

    fn check(ra_fixture: &str, expect: expect_test::Expect) {
        let (analysis, file_id) = fixture::file(ra_fixture);
        let syn = analysis.syntax_tree(file_id, None).unwrap();
        expect.assert_eq(&syn)
    }
    fn check_range(ra_fixture: &str, expect: expect_test::Expect) {
        let (analysis, frange) = fixture::range(ra_fixture);
        let syn = analysis.syntax_tree(frange.file_id, Some(frange.range)).unwrap();
        expect.assert_eq(&syn)
    }

    #[test]
    fn test_syntax_tree_without_range() {
        // Basic syntax
        check(
            r#"fn foo() {}"#,
            expect![[r#"
                SOURCE_FILE@0..11
                  FN@0..11
                    FN_KW@0..2 "fn"
                    WHITESPACE@2..3 " "
                    NAME@3..6
                      IDENT@3..6 "foo"
                    PARAM_LIST@6..8
                      L_PAREN@6..7 "("
                      R_PAREN@7..8 ")"
                    WHITESPACE@8..9 " "
                    BLOCK_EXPR@9..11
                      STMT_LIST@9..11
                        L_CURLY@9..10 "{"
                        R_CURLY@10..11 "}"
            "#]],
        );

        check(
            r#"
fn test() {
    assert!("
    fn foo() {
    }
    ", "");
}"#,
            expect![[r#"
                SOURCE_FILE@0..60
                  FN@0..60
                    FN_KW@0..2 "fn"
                    WHITESPACE@2..3 " "
                    NAME@3..7
                      IDENT@3..7 "test"
                    PARAM_LIST@7..9
                      L_PAREN@7..8 "("
                      R_PAREN@8..9 ")"
                    WHITESPACE@9..10 " "
                    BLOCK_EXPR@10..60
                      STMT_LIST@10..60
                        L_CURLY@10..11 "{"
                        WHITESPACE@11..16 "\n    "
                        EXPR_STMT@16..58
                          MACRO_EXPR@16..57
                            MACRO_CALL@16..57
                              PATH@16..22
                                PATH_SEGMENT@16..22
                                  NAME_REF@16..22
                                    IDENT@16..22 "assert"
                              BANG@22..23 "!"
                              TOKEN_TREE@23..57
                                L_PAREN@23..24 "("
                                STRING@24..52 "\"\n    fn foo() {\n     ..."
                                COMMA@52..53 ","
                                WHITESPACE@53..54 " "
                                STRING@54..56 "\"\""
                                R_PAREN@56..57 ")"
                          SEMICOLON@57..58 ";"
                        WHITESPACE@58..59 "\n"
                        R_CURLY@59..60 "}"
            "#]],
        )
    }

    #[test]
    fn test_syntax_tree_with_range() {
        check_range(
            r#"$0fn foo() {}$0"#,
            expect![[r#"
                FN@0..11
                  FN_KW@0..2 "fn"
                  WHITESPACE@2..3 " "
                  NAME@3..6
                    IDENT@3..6 "foo"
                  PARAM_LIST@6..8
                    L_PAREN@6..7 "("
                    R_PAREN@7..8 ")"
                  WHITESPACE@8..9 " "
                  BLOCK_EXPR@9..11
                    STMT_LIST@9..11
                      L_CURLY@9..10 "{"
                      R_CURLY@10..11 "}"
            "#]],
        );

        check_range(
            r#"
fn test() {
    $0assert!("
    fn foo() {
    }
    ", "");$0
}"#,
            expect![[r#"
                EXPR_STMT@16..58
                  MACRO_EXPR@16..57
                    MACRO_CALL@16..57
                      PATH@16..22
                        PATH_SEGMENT@16..22
                          NAME_REF@16..22
                            IDENT@16..22 "assert"
                      BANG@22..23 "!"
                      TOKEN_TREE@23..57
                        L_PAREN@23..24 "("
                        STRING@24..52 "\"\n    fn foo() {\n     ..."
                        COMMA@52..53 ","
                        WHITESPACE@53..54 " "
                        STRING@54..56 "\"\""
                        R_PAREN@56..57 ")"
                  SEMICOLON@57..58 ";"
            "#]],
        );
    }

    #[test]
    fn test_syntax_tree_inside_string() {
        check_range(
            r#"fn test() {
    assert!("
$0fn foo() {
}$0
fn bar() {
}
    ", "");
}"#,
            expect![[r#"
                SOURCE_FILE@0..12
                  FN@0..12
                    FN_KW@0..2 "fn"
                    WHITESPACE@2..3 " "
                    NAME@3..6
                      IDENT@3..6 "foo"
                    PARAM_LIST@6..8
                      L_PAREN@6..7 "("
                      R_PAREN@7..8 ")"
                    WHITESPACE@8..9 " "
                    BLOCK_EXPR@9..12
                      STMT_LIST@9..12
                        L_CURLY@9..10 "{"
                        WHITESPACE@10..11 "\n"
                        R_CURLY@11..12 "}"
            "#]],
        );

        // With a raw string
        check_range(
            r###"fn test() {
    assert!(r#"
$0fn foo() {
}$0
fn bar() {
}
    "#, "");
}"###,
            expect![[r#"
                SOURCE_FILE@0..12
                  FN@0..12
                    FN_KW@0..2 "fn"
                    WHITESPACE@2..3 " "
                    NAME@3..6
                      IDENT@3..6 "foo"
                    PARAM_LIST@6..8
                      L_PAREN@6..7 "("
                      R_PAREN@7..8 ")"
                    WHITESPACE@8..9 " "
                    BLOCK_EXPR@9..12
                      STMT_LIST@9..12
                        L_CURLY@9..10 "{"
                        WHITESPACE@10..11 "\n"
                        R_CURLY@11..12 "}"
            "#]],
        );

        // With a raw string
        check_range(
            r###"fn test() {
    assert!(r$0#"
fn foo() {
}
fn bar() {
}"$0#, "");
}"###,
            expect![[r#"
                SOURCE_FILE@0..25
                  FN@0..12
                    FN_KW@0..2 "fn"
                    WHITESPACE@2..3 " "
                    NAME@3..6
                      IDENT@3..6 "foo"
                    PARAM_LIST@6..8
                      L_PAREN@6..7 "("
                      R_PAREN@7..8 ")"
                    WHITESPACE@8..9 " "
                    BLOCK_EXPR@9..12
                      STMT_LIST@9..12
                        L_CURLY@9..10 "{"
                        WHITESPACE@10..11 "\n"
                        R_CURLY@11..12 "}"
                  WHITESPACE@12..13 "\n"
                  FN@13..25
                    FN_KW@13..15 "fn"
                    WHITESPACE@15..16 " "
                    NAME@16..19
                      IDENT@16..19 "bar"
                    PARAM_LIST@19..21
                      L_PAREN@19..20 "("
                      R_PAREN@20..21 ")"
                    WHITESPACE@21..22 " "
                    BLOCK_EXPR@22..25
                      STMT_LIST@22..25
                        L_CURLY@22..23 "{"
                        WHITESPACE@23..24 "\n"
                        R_CURLY@24..25 "}"
            "#]],
        );
    }
}
