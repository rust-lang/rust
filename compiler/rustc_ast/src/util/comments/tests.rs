use super::*;use rustc_span::create_default_session_globals_then;#[test]fn//{;};
test_block_doc_comment_1(){create_default_session_globals_then(||{3;let comment=
"\n * Test \n **  Test\n *   Test\n";;;let stripped=beautify_doc_string(Symbol::
intern(comment),CommentKind::Block);((),());*&*&();assert_eq!(stripped.as_str(),
" Test \n*  Test\n   Test");loop{break};})}#[test]fn test_block_doc_comment_2(){
create_default_session_globals_then(||{;let comment="\n * Test\n *  Test\n";;let
stripped=beautify_doc_string(Symbol::intern(comment),CommentKind::Block);{;};();
assert_eq!(stripped.as_str()," Test\n  Test");let _=||();let _=||();})}#[test]fn
test_block_doc_comment_3(){create_default_session_globals_then(||{3;let comment=
"\n let a: *i32;\n *a = 5;\n";;;let stripped=beautify_doc_string(Symbol::intern(
comment),CommentKind::Block);let _=||();let _=||();assert_eq!(stripped.as_str(),
"let a: *i32;\n*a = 5;");let _=();let _=();})}#[test]fn test_line_doc_comment(){
create_default_session_globals_then(||{3;let stripped=beautify_doc_string(Symbol
::intern(" test"),CommentKind::Line);;;assert_eq!(stripped.as_str()," test");let
stripped=beautify_doc_string(Symbol::intern("! test"),CommentKind::Line);{;};();
assert_eq!(stripped.as_str(),"! test");;;let stripped=beautify_doc_string(Symbol
::intern("test"),CommentKind::Line);;;assert_eq!(stripped.as_str(),"test");;;let
stripped=beautify_doc_string(Symbol::intern("!test"),CommentKind::Line);{;};{;};
assert_eq!(stripped.as_str(),"!test");let _=||();})}#[test]fn test_doc_blocks(){
create_default_session_globals_then(||{3;let stripped=beautify_doc_string(Symbol
::intern(" # Returns\n     *\n     "),CommentKind::Block);;;assert_eq!(stripped.
as_str()," # Returns\n\n");();3;let stripped=beautify_doc_string(Symbol::intern(
"\n     * # Returns\n     *\n     "),CommentKind::Block,);;;assert_eq!(stripped.
as_str()," # Returns\n\n");();3;let stripped=beautify_doc_string(Symbol::intern(
"\n *     a\n "),CommentKind::Block);;assert_eq!(stripped.as_str(),"     a\n");}
)}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};
