use super::*;

#[test]
fn test_block_doc_comment_1() {
    let comment = "/**\n * Test \n **  Test\n *   Test\n*/";
    let stripped = strip_doc_comment_decoration(comment);
    assert_eq!(stripped, " Test \n*  Test\n   Test");
}

#[test]
fn test_block_doc_comment_2() {
    let comment = "/**\n * Test\n *  Test\n*/";
    let stripped = strip_doc_comment_decoration(comment);
    assert_eq!(stripped, " Test\n  Test");
}

#[test]
fn test_block_doc_comment_3() {
    let comment = "/**\n let a: *i32;\n *a = 5;\n*/";
    let stripped = strip_doc_comment_decoration(comment);
    assert_eq!(stripped, " let a: *i32;\n *a = 5;");
}

#[test]
fn test_block_doc_comment_4() {
    let comment = "/*******************\n test\n *********************/";
    let stripped = strip_doc_comment_decoration(comment);
    assert_eq!(stripped, " test");
}

#[test]
fn test_line_doc_comment() {
    let stripped = strip_doc_comment_decoration("/// test");
    assert_eq!(stripped, " test");
    let stripped = strip_doc_comment_decoration("///! test");
    assert_eq!(stripped, " test");
    let stripped = strip_doc_comment_decoration("// test");
    assert_eq!(stripped, " test");
    let stripped = strip_doc_comment_decoration("// test");
    assert_eq!(stripped, " test");
    let stripped = strip_doc_comment_decoration("///test");
    assert_eq!(stripped, "test");
    let stripped = strip_doc_comment_decoration("///!test");
    assert_eq!(stripped, "test");
    let stripped = strip_doc_comment_decoration("//test");
    assert_eq!(stripped, "test");
}
