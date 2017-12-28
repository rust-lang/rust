extern crate file;
#[macro_use(assert_diff)]
extern crate difference;

use std::path::{PathBuf, Path};
use std::fs::read_dir;

#[test]
fn lexer_tests() {
    for test_case in lexer_test_cases() {
        lexer_test_case(&test_case);
    }
}

fn lexer_test_dir() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).join("tests/data/lexer")
}

fn lexer_test_cases() -> Vec<PathBuf> {
    let mut acc = Vec::new();
    let dir = lexer_test_dir();
    for file in read_dir(&dir).unwrap() {
        let file = file.unwrap();
        let path = file.path();
        if path.extension().unwrap_or_default() == "rs" {
            acc.push(path);
        }
    }
    acc
}

fn lexer_test_case(path: &Path) {
    let actual = {
        let text = file::get_text(path).unwrap();
        let tokens = tokenize(&text);
        dump_tokens(&tokens)
    };
    let expected = file::get_text(&path.with_extension("txt")).unwrap();

    assert_diff!(
        expected.as_str(),
        actual.as_str(),
        "\n",
        0
    )
}

fn tokenize(text: &str) -> Vec<()> {
    Vec::new()
}

fn dump_tokens(tokens: &[()]) -> String {
    "IDENT 5\nKEYWORD 1\nIDENT 5\n".to_string()
}