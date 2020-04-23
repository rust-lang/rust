// run-pass

#![feature(proc_macro_hygiene, proc_macro_is_available)]

extern crate proc_macro;

// aux-build:is-available.rs
extern crate is_available;

fn main() {
    let a = proc_macro::is_available();
    let b = is_available::from_inside_proc_macro!();
    let c = proc_macro::is_available();
    assert!(!a);
    assert!(b);
    assert!(!c);
}
