use super::*;
use crate::with_default_session_globals;

#[test]
fn test_block_doc_comment_1() {
    with_default_session_globals(|| {
        let comment = "/**\n * Test \n **  Test\n *   Test\n*/";
        let stripped = strip_doc_comment_decoration(Symbol::intern(comment));
        assert_eq!(stripped, " Test \n*  Test\n   Test");
    })
}

#[test]
fn test_block_doc_comment_2() {
    with_default_session_globals(|| {
        let comment = "/**\n * Test\n *  Test\n*/";
        let stripped = strip_doc_comment_decoration(Symbol::intern(comment));
        assert_eq!(stripped, " Test\n  Test");
    })
}

#[test]
fn test_block_doc_comment_3() {
    with_default_session_globals(|| {
        let comment = "/**\n let a: *i32;\n *a = 5;\n*/";
        let stripped = strip_doc_comment_decoration(Symbol::intern(comment));
        assert_eq!(stripped, " let a: *i32;\n *a = 5;");
    })
}

#[test]
fn test_block_doc_comment_4() {
    with_default_session_globals(|| {
        let comment = "/*******************\n test\n *********************/";
        let stripped = strip_doc_comment_decoration(Symbol::intern(comment));
        assert_eq!(stripped, " test");
    })
}

#[test]
fn test_line_doc_comment() {
    with_default_session_globals(|| {
        let stripped = strip_doc_comment_decoration(Symbol::intern("/// test"));
        assert_eq!(stripped, " test");
        let stripped = strip_doc_comment_decoration(Symbol::intern("///! test"));
        assert_eq!(stripped, " test");
        let stripped = strip_doc_comment_decoration(Symbol::intern("// test"));
        assert_eq!(stripped, " test");
        let stripped = strip_doc_comment_decoration(Symbol::intern("// test"));
        assert_eq!(stripped, " test");
        let stripped = strip_doc_comment_decoration(Symbol::intern("///test"));
        assert_eq!(stripped, "test");
        let stripped = strip_doc_comment_decoration(Symbol::intern("///!test"));
        assert_eq!(stripped, "test");
        let stripped = strip_doc_comment_decoration(Symbol::intern("//test"));
        assert_eq!(stripped, "test");
    })
}
