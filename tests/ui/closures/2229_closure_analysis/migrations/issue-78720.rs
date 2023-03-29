// run-pass

#![warn(rust_2021_incompatible_closure_captures)]
#![allow(drop_ref, drop_copy)]

fn main() {
    if let a = "" {
        //~^ WARNING: irrefutable `if let` pattern
        drop(|_: ()| drop(a));
    }
}
