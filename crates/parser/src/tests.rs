use std::{
    fmt::Write,
    fs,
    path::{Path, PathBuf},
};

use expect_test::expect_file;

use crate::LexedStr;

#[test]
fn valid_lexes_input() {
    for case in TestCase::list("lexer/ok") {
        let actual = lex(&case.text);
        expect_file![case.txt].assert_eq(&actual)
    }
}

#[test]
fn invalid_lexes_input() {
    for case in TestCase::list("lexer/err") {
        let actual = lex(&case.text);
        expect_file![case.txt].assert_eq(&actual)
    }
}

fn lex(text: &str) -> String {
    let lexed = LexedStr::new(text);

    let mut res = String::new();
    for i in 0..lexed.len() {
        let kind = lexed.kind(i);
        let text = lexed.text(i);
        let error = lexed.error(i);

        let error = error.map(|err| format!(" error: {}", err)).unwrap_or_default();
        writeln!(res, "{:?} {:?}{}", kind, text, error).unwrap();
    }
    res
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct TestCase {
    rs: PathBuf,
    txt: PathBuf,
    text: String,
}

impl TestCase {
    fn list(path: &'static str) -> Vec<TestCase> {
        let crate_root_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let test_data_dir = crate_root_dir.join("test_data");
        let dir = test_data_dir.join(path);

        let mut res = Vec::new();
        let read_dir = fs::read_dir(&dir)
            .unwrap_or_else(|err| panic!("can't `read_dir` {}: {}", dir.display(), err));
        for file in read_dir {
            let file = file.unwrap();
            let path = file.path();
            if path.extension().unwrap_or_default() == "rs" {
                let rs = path;
                let txt = rs.with_extension("txt");
                let text = fs::read_to_string(&rs).unwrap();
                res.push(TestCase { rs, txt, text });
            }
        }
        res.sort();
        res
    }
}
