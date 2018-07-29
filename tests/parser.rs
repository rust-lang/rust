extern crate libsyntax2;
extern crate testutils;

use libsyntax2::{parse};
use libsyntax2::utils::{dump_tree_green};
use testutils::dir_tests;

#[test]
fn parser_tests() {
    dir_tests(&["parser/inline", "parser/ok", "parser/err"], |text| {
        let file = parse(text.to_string());
        dump_tree_green(&file)
    })
}
