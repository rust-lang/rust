use xtask::{
    codegen::{self, Mode},
    run_rustfmt,
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
