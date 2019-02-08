use tools::{generate, gen_tests, run_rustfmt, Verify};

#[test]
fn generated_grammar_is_fresh() {
    if let Err(error) = generate(Verify) {
        panic!("{}. Please update it by running `cargo gen-syntax`", error);
    }
}

#[test]
fn generated_tests_are_fresh() {
    if let Err(error) = gen_tests(Verify) {
        panic!("{}. Please update tests by running `cargo gen-tests`", error);
    }
}

#[test]
fn check_code_formatting() {
    if let Err(error) = run_rustfmt(Verify) {
        panic!("{}. Please format the code by running `cargo format`", error);
    }
}
