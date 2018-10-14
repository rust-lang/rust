extern crate tools;

use std::path::Path;
use tools::{render_template, update};

const GRAMMAR: &str = "../ra_syntax/src/grammar.ron";
const SYNTAX_KINDS: &str = "../ra_syntax/src/syntax_kinds/generated.rs";
const SYNTAX_KINDS_TEMPLATE: &str = "../ra_syntax/src/syntax_kinds/generated.rs.tera";
const AST: &str = "../ra_syntax/src/ast/generated.rs";
const AST_TEMPLATE: &str = "../ra_syntax/src/ast/generated.rs.tera";

#[test]
fn verify_template_generation() {
    if let Err(error) = update(Path::new(SYNTAX_KINDS), &render_template(SYNTAX_KINDS_TEMPLATE, GRAMMAR).unwrap(), true) {
        panic!("{}. Please update it by running `cargo gen-kinds`", error);
    }
    if let Err(error) = update(Path::new(AST), &render_template(AST_TEMPLATE, GRAMMAR).unwrap(), true) {
        panic!("{}. Please update it by running `cargo gen-kinds`", error);
    }
}