extern crate libsyntax2;
extern crate testutils;

use libsyntax2::{tokenize, parse_green};
use libsyntax2::utils::{dump_tree_green};
use testutils::dir_tests;

#[test]
fn parser_tests() {
    dir_tests(&["parser/inline", "parser/ok", "parser/err"], |text| {
        let tokens = tokenize(text);
        let file = parse_green(text.to_string(), &tokens);
        dump_tree_green(&file)
    })
}
