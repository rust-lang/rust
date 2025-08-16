//@ reference: input.crlf
//@ ignore-cross-compile

use run_make_support::{cwd, rfs, run, rustc};

fn main() {
    let test_content = "/// Doc comment that ends in CRLF\r\n\
pub fn foo() {}\r\n\
\r\n\
/** Block doc comment that\r\n\
 *  contains CRLF characters\r\n\
 */\r\n\
pub fn bar() {}\r\n\
\r\n\
fn main() {\r\n\
    let s = \"string\r\nliteral\";\r\n\
    assert_eq!(s, \"string\\nliteral\");\r\n\
\r\n\
    let s = \"literal with \\\r\n\
             escaped newline\";\r\n\
    assert_eq!(s, \"literal with escaped newline\");\r\n\
\r\n\
    let s = r\"string\r\nliteral\";\r\n\
    assert_eq!(s, \"string\\nliteral\");\r\n\
    let s = br\"byte string\r\nliteral\";\r\n\
    assert_eq!(s, \"byte string\\nliteral\".as_bytes());\r\n\
\r\n\
    // validate that our source file has CRLF endings\r\n\
    let source = include_str!(file!());\r\n\
    assert!(source.contains(\"string\\r\\nliteral\"));\r\n\
}\r\n";

    let test_path = cwd().join("lexer-crlf-line-endings-string-literal-doc-comment.rs");
    rfs::write(&test_path, test_content);

    let test_bytes = rfs::read(&test_path);
    assert!(test_bytes.windows(2).any(|w| w == b"\r\n"), "File must contain CRLF line endings");

    rustc().input(&test_path).run();
}
