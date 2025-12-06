//@ needs-asm-support

use run_make_support::symbols::object_contains_any_symbol;
use run_make_support::{bin_name, rustc};

fn main() {
    rustc().input("main.rs").opt().run();
    let mut unused = vec!["unused", "unused_link_section"];
    assert!(!object_contains_any_symbol(bin_name("main"), &unused));
}
