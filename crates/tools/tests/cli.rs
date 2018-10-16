extern crate tools;

use tools::{
    generate, Verify
};

#[test]
fn verify_template_generation() {
    if let Err(error) = generate(Verify) {
        panic!("{}. Please update it by running `cargo gen-kinds`", error);
    }
}
