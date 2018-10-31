extern crate tools;

use tools::{
    generate, Verify, run_rustfmt,
};

#[test]
fn verify_template_generation() {
    if let Err(error) = generate(Verify) {
        panic!("{}. Please update it by running `cargo gen-syntax`", error);
    }
}

#[test]
fn check_code_formatting() {
    if let Err(error) = run_rustfmt(Verify) {
        panic!("{}. Please format the code by running `cargo format`", error);
    }
}
