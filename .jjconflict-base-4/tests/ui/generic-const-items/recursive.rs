//@ build-fail

#![feature(generic_const_items)]
#![allow(incomplete_features)]
#![recursion_limit = "15"]

const RECUR<T>: () = RECUR::<(T,)>;

fn main() {
    let _ = RECUR::<()>; //~ ERROR: queries overflow the depth limit!
}
