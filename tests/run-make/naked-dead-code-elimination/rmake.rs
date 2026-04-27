//@ ignore-cross-compile
//@ needs-asm-support
//@ needs-asm-mnemonic:RET

use run_make_support::symbols::object_contains_any_symbol;
use run_make_support::{bin_name, rustc};

fn main() {
    let bin = bin_name("main");

    rustc().input("main.rs").opt().function_sections(false).run();

    // Check that the naked symbol is eliminated when the "clothed" one is.

    assert_eq!(
        object_contains_any_symbol(&bin, &["unused_clothed"]),
        object_contains_any_symbol(&bin, &["unused"])
    );

    assert_eq!(
        object_contains_any_symbol(&bin, &["unused_link_section_clothed"]),
        object_contains_any_symbol(&bin, &["unused_link_section"])
    );

    // ---

    rustc().input("main.rs").opt().function_sections(true).run();

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
