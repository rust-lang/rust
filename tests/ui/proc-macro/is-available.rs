//@ run-pass

extern crate proc_macro;

//@ proc-macro: is-available.rs
//@ ignore-backends: gcc
extern crate is_available;

fn main() {
    let a = proc_macro::is_available();
    let b = is_available::from_inside_proc_macro!();
    let c = proc_macro::is_available();
    assert!(!a);
    assert!(b);
    assert!(!c);
}
