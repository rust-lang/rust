use walkdir::WalkDir;
use xtask::{
    codegen::{self, Mode},
    project_root, run_rustfmt,
};

#[test]
fn generated_grammar_is_fresh() {
    if let Err(error) = codegen::generate_syntax(Mode::Verify) {
        panic!("{}. Please update it by running `cargo xtask codegen`", error);
    }
}

#[test]
fn generated_tests_are_fresh() {
    if let Err(error) = codegen::generate_parser_tests(Mode::Verify) {
        panic!("{}. Please update tests by running `cargo xtask codegen`", error);
    }
}

#[test]
fn generated_assists_are_fresh() {
    if let Err(error) = codegen::generate_assists_docs(Mode::Verify) {
        panic!("{}. Please update assists by running `cargo xtask codegen`", error);
    }
}

#[test]
fn check_code_formatting() {
    if let Err(error) = run_rustfmt(Mode::Verify) {
        panic!("{}. Please format the code by running `cargo format`", error);
    }
}

#[test]
fn no_todo() {
    WalkDir::new(project_root().join("crates")).into_iter().for_each(|e| {
        let e = e.unwrap();
        if e.path().extension().map(|it| it != "rs").unwrap_or(true) {
            return;
        }
        if e.path().ends_with("tests/cli.rs") {
            return;
        }
        let text = std::fs::read_to_string(e.path()).unwrap();
        if text.contains("TODO") || text.contains("TOOD") {
            panic!(
                "\nTODO markers should not be committed to the master branch,\n\
                 use FIXME instead\n\
                 {}\n",
                e.path().display(),
            )
        }
    })
}
