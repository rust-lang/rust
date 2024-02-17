//@ run-pass

#![warn(rust_2021_incompatible_closure_captures)]
#![allow(dropping_references, dropping_copy_types)]

fn main() {
    if let a = "" {
        //~^ WARNING: irrefutable `if let` pattern
        drop(|_: ()| drop(a));
    }
}
