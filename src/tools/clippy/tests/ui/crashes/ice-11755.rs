//@ check-pass

#![warn(clippy::unused_enumerate_index)]

fn main() {
    for () in [()].iter() {}
}
