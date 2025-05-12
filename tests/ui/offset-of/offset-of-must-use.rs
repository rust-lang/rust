//@ check-pass

#![warn(unused)]

fn main() {
    core::mem::offset_of!((String,), 0);
    //~^ WARN unused `offset_of` call that must be used
}
