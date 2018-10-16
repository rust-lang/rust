extern crate tools;

use tools::{AST, AST_TEMPLATE, SYNTAX_KINDS, SYNTAX_KINDS_TEMPLATE, render_template, update, project_root};

#[test]
fn verify_template_generation() {
    if let Err(error) = update(&project_root().join(SYNTAX_KINDS), &render_template(&project_root().join(SYNTAX_KINDS_TEMPLATE)).unwrap(), true) {
        panic!("{}. Please update it by running `cargo gen-kinds`", error);
    }
    if let Err(error) = update(&project_root().join(AST), &render_template(&project_root().join(AST_TEMPLATE)).unwrap(), true) {
        panic!("{}. Please update it by running `cargo gen-kinds`", error);
    }
}
