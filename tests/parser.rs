extern crate file;
extern crate libsyntax2;
extern crate testutils;

use libsyntax2::{tokenize, parse};
use libsyntax2::utils::dump_tree;
use testutils::dir_tests;

#[test]
fn parser_tests() {
    dir_tests(
        &["parser/ok", "parser/err"],
        |text| {
            let tokens = tokenize(text);
            let file = parse(text.to_string(), &tokens);
            dump_tree(&file)
        }
    )
}
