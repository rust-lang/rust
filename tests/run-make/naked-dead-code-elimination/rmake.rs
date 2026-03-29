//@ needs-asm-support

use run_make_support::symbols::object_contains_any_symbol;
use run_make_support::{bin_name, rustc};

fn main() {
    rustc().input("main.rs").opt().function_sections(true).run();

    let bin = bin_name("main");

    // Check that the naked symbol is eliminated when the "clothed" one is.

    assert_eq!(
        object_contains_any_symbol(&bin, &["unused_clothed"]),
        object_contains_any_symbol(&bin, &["unused"])
    );

    assert_eq!(
        object_contains_any_symbol(&bin, &["unused_link_section_clothed"]),
        object_contains_any_symbol(&bin, &["unused_link_section"])
    );
}
