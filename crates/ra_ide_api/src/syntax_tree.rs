use crate::db::RootDatabase;
use ra_db::SourceDatabase;
use ra_syntax::{
    algo, AstNode, NodeOrToken, SourceFile,
    SyntaxKind::{RAW_STRING, STRING},
    SyntaxToken, TextRange,
};

pub use ra_db::FileId;

pub(crate) fn syntax_tree(
    db: &RootDatabase,
    file_id: FileId,
    text_range: Option<TextRange>,
) -> String {
    let parse = db.parse(file_id);
    if let Some(text_range) = text_range {
        let node = match algo::find_covering_element(parse.tree().syntax(), text_range) {
            NodeOrToken::Node(node) => node,
            NodeOrToken::Token(token) => {
                if let Some(tree) = syntax_tree_for_string(&token, text_range) {
                    return tree;
                }
                token.parent()
            }
        };

        format!("{:#?}", node)
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
        STRING | RAW_STRING => syntax_tree_for_token(token, text_range),
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
        return Some(format!("{:#?}", parsed.tree().syntax()));
    }

    None
}

#[cfg(test)]
mod tests {
    use test_utils::assert_eq_text;

    use crate::mock_analysis::{single_file, single_file_with_range};

    #[test]
    fn test_syntax_tree_without_range() {
        // Basic syntax
        let (analysis, file_id) = single_file(r#"fn foo() {}"#);
        let syn = analysis.syntax_tree(file_id, None).unwrap();

        assert_eq_text!(
            syn.trim(),
            r#"
SOURCE_FILE@[0; 11)
  FN_DEF@[0; 11)
    FN_KW@[0; 2) "fn"
    WHITESPACE@[2; 3) " "
    NAME@[3; 6)
      IDENT@[3; 6) "foo"
    PARAM_LIST@[6; 8)
      L_PAREN@[6; 7) "("
      R_PAREN@[7; 8) ")"
    WHITESPACE@[8; 9) " "
    BLOCK@[9; 11)
      L_CURLY@[9; 10) "{"
      R_CURLY@[10; 11) "}"
"#
            .trim()
        );

        let (analysis, file_id) = single_file(
            r#"
fn test() {
    assert!("
    fn foo() {
    }
    ", "");
}"#
            .trim(),
        );
        let syn = analysis.syntax_tree(file_id, None).unwrap();

        assert_eq_text!(
            syn.trim(),
            r#"
SOURCE_FILE@[0; 60)
  FN_DEF@[0; 60)
    FN_KW@[0; 2) "fn"
    WHITESPACE@[2; 3) " "
    NAME@[3; 7)
      IDENT@[3; 7) "test"
    PARAM_LIST@[7; 9)
      L_PAREN@[7; 8) "("
      R_PAREN@[8; 9) ")"
    WHITESPACE@[9; 10) " "
    BLOCK@[10; 60)
      L_CURLY@[10; 11) "{"
      WHITESPACE@[11; 16) "\n    "
      EXPR_STMT@[16; 58)
        MACRO_CALL@[16; 57)
          PATH@[16; 22)
            PATH_SEGMENT@[16; 22)
              NAME_REF@[16; 22)
                IDENT@[16; 22) "assert"
          EXCL@[22; 23) "!"
          TOKEN_TREE@[23; 57)
            L_PAREN@[23; 24) "("
            STRING@[24; 52) "\"\n    fn foo() {\n     ..."
            COMMA@[52; 53) ","
            WHITESPACE@[53; 54) " "
            STRING@[54; 56) "\"\""
            R_PAREN@[56; 57) ")"
        SEMI@[57; 58) ";"
      WHITESPACE@[58; 59) "\n"
      R_CURLY@[59; 60) "}"
"#
            .trim()
        );
    }

    #[test]
    fn test_syntax_tree_with_range() {
        let (analysis, range) = single_file_with_range(r#"<|>fn foo() {}<|>"#.trim());
        let syn = analysis.syntax_tree(range.file_id, Some(range.range)).unwrap();

        assert_eq_text!(
            syn.trim(),
            r#"
FN_DEF@[0; 11)
  FN_KW@[0; 2) "fn"
  WHITESPACE@[2; 3) " "
  NAME@[3; 6)
    IDENT@[3; 6) "foo"
  PARAM_LIST@[6; 8)
    L_PAREN@[6; 7) "("
    R_PAREN@[7; 8) ")"
  WHITESPACE@[8; 9) " "
  BLOCK@[9; 11)
    L_CURLY@[9; 10) "{"
    R_CURLY@[10; 11) "}"
"#
            .trim()
        );

        let (analysis, range) = single_file_with_range(
            r#"fn test() {
    <|>assert!("
    fn foo() {
    }
    ", "");<|>
}"#
            .trim(),
        );
        let syn = analysis.syntax_tree(range.file_id, Some(range.range)).unwrap();

        assert_eq_text!(
            syn.trim(),
            r#"
EXPR_STMT@[16; 58)
  MACRO_CALL@[16; 57)
    PATH@[16; 22)
      PATH_SEGMENT@[16; 22)
        NAME_REF@[16; 22)
          IDENT@[16; 22) "assert"
    EXCL@[22; 23) "!"
    TOKEN_TREE@[23; 57)
      L_PAREN@[23; 24) "("
      STRING@[24; 52) "\"\n    fn foo() {\n     ..."
      COMMA@[52; 53) ","
      WHITESPACE@[53; 54) " "
      STRING@[54; 56) "\"\""
      R_PAREN@[56; 57) ")"
  SEMI@[57; 58) ";"
"#
            .trim()
        );
    }

    #[test]
    fn test_syntax_tree_inside_string() {
        let (analysis, range) = single_file_with_range(
            r#"fn test() {
    assert!("
<|>fn foo() {
}<|>
fn bar() {
}
    ", "");
}"#
            .trim(),
        );
        let syn = analysis.syntax_tree(range.file_id, Some(range.range)).unwrap();
        assert_eq_text!(
            syn.trim(),
            r#"
SOURCE_FILE@[0; 12)
  FN_DEF@[0; 12)
    FN_KW@[0; 2) "fn"
    WHITESPACE@[2; 3) " "
    NAME@[3; 6)
      IDENT@[3; 6) "foo"
    PARAM_LIST@[6; 8)
      L_PAREN@[6; 7) "("
      R_PAREN@[7; 8) ")"
    WHITESPACE@[8; 9) " "
    BLOCK@[9; 12)
      L_CURLY@[9; 10) "{"
      WHITESPACE@[10; 11) "\n"
      R_CURLY@[11; 12) "}"
"#
            .trim()
        );

        // With a raw string
        let (analysis, range) = single_file_with_range(
            r###"fn test() {
    assert!(r#"
<|>fn foo() {
}<|>
fn bar() {
}
    "#, "");
}"###
                .trim(),
        );
        let syn = analysis.syntax_tree(range.file_id, Some(range.range)).unwrap();
        assert_eq_text!(
            syn.trim(),
            r#"
SOURCE_FILE@[0; 12)
  FN_DEF@[0; 12)
    FN_KW@[0; 2) "fn"
    WHITESPACE@[2; 3) " "
    NAME@[3; 6)
      IDENT@[3; 6) "foo"
    PARAM_LIST@[6; 8)
      L_PAREN@[6; 7) "("
      R_PAREN@[7; 8) ")"
    WHITESPACE@[8; 9) " "
    BLOCK@[9; 12)
      L_CURLY@[9; 10) "{"
      WHITESPACE@[10; 11) "\n"
      R_CURLY@[11; 12) "}"
"#
            .trim()
        );

        // With a raw string
        let (analysis, range) = single_file_with_range(
            r###"fn test() {
    assert!(r<|>#"
fn foo() {
}
fn bar() {
}"<|>#, "");
}"###
                .trim(),
        );
        let syn = analysis.syntax_tree(range.file_id, Some(range.range)).unwrap();
        assert_eq_text!(
            syn.trim(),
            r#"
SOURCE_FILE@[0; 25)
  FN_DEF@[0; 12)
    FN_KW@[0; 2) "fn"
    WHITESPACE@[2; 3) " "
    NAME@[3; 6)
      IDENT@[3; 6) "foo"
    PARAM_LIST@[6; 8)
      L_PAREN@[6; 7) "("
      R_PAREN@[7; 8) ")"
    WHITESPACE@[8; 9) " "
    BLOCK@[9; 12)
      L_CURLY@[9; 10) "{"
      WHITESPACE@[10; 11) "\n"
      R_CURLY@[11; 12) "}"
  WHITESPACE@[12; 13) "\n"
  FN_DEF@[13; 25)
    FN_KW@[13; 15) "fn"
    WHITESPACE@[15; 16) " "
    NAME@[16; 19)
      IDENT@[16; 19) "bar"
    PARAM_LIST@[19; 21)
      L_PAREN@[19; 20) "("
      R_PAREN@[20; 21) ")"
    WHITESPACE@[21; 22) " "
    BLOCK@[22; 25)
      L_CURLY@[22; 23) "{"
      WHITESPACE@[23; 24) "\n"
      R_CURLY@[24; 25) "}"
"#
            .trim()
        );
    }
}
