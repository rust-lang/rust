use super::*;
use rustc_span::with_default_session_globals;

#[test]
fn test_block_doc_comment_1() {
    with_default_session_globals(|| {
        let comment = "\n * Test \n **  Test\n *   Test\n";
        let stripped = beautify_doc_string(Symbol::intern(comment));
        assert_eq!(stripped.as_str(), " Test \n*  Test\n   Test");
    })
}

#[test]
fn test_block_doc_comment_2() {
    with_default_session_globals(|| {
        let comment = "\n * Test\n *  Test\n";
        let stripped = beautify_doc_string(Symbol::intern(comment));
        assert_eq!(stripped.as_str(), " Test\n  Test");
    })
}

#[test]
fn test_block_doc_comment_3() {
    with_default_session_globals(|| {
        let comment = "\n let a: *i32;\n *a = 5;\n";
        let stripped = beautify_doc_string(Symbol::intern(comment));
        assert_eq!(stripped.as_str(), " let a: *i32;\n *a = 5;");
    })
}

#[test]
fn test_line_doc_comment() {
    with_default_session_globals(|| {
        let stripped = beautify_doc_string(Symbol::intern(" test"));
        assert_eq!(stripped.as_str(), " test");
        let stripped = beautify_doc_string(Symbol::intern("! test"));
        assert_eq!(stripped.as_str(), "! test");
        let stripped = beautify_doc_string(Symbol::intern("test"));
        assert_eq!(stripped.as_str(), "test");
        let stripped = beautify_doc_string(Symbol::intern("!test"));
        assert_eq!(stripped.as_str(), "!test");
    })
}
