//@normalize-stderr-test: "\| +\^+" -> "| ^"
//@normalize-stderr-test: "\n  +[0-9]+:[^\n]+" -> "$1"
//@normalize-stderr-test: "\n at [^\n]+" -> "$1"

// Checks that nested panics work correctly.

use std::panic::catch_unwind;

fn double() {
    struct Double;

    impl Drop for Double {
        fn drop(&mut self) {
            let _ = catch_unwind(|| panic!("twice"));
        }
    }

    let _d = Double;

    panic!("once");
}

fn main() {
    assert!(catch_unwind(|| double()).is_err());
}
