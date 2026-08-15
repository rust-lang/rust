//@ edition: 2024
//@ aux-build: aux_with_props.rs
//@ compile-flags: --check-cfg=cfg(this_is_aux)
//@ run-pass

// Test that auxiliaries are built using the directives in the auxiliary file,
// and don't just accidentally use the directives of the main test file.

extern crate aux_with_props;

fn main() {
    assert!(!cfg!(this_is_aux));
    assert!(aux_with_props::aux_directives_are_respected());
}
