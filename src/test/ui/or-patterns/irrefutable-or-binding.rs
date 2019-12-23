// run-pass

#![feature(or_patterns)]
//~^ WARN the feature `or_patterns` is incomplete and may cause the compiler to crash

fn main() {
    let _ | _ = (); // ok
}
