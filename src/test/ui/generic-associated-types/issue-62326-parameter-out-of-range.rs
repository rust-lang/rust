#![allow(incomplete_features)]
#![feature(generic_associated_types)]

// check-pass

trait Iterator {
    type Item<'a>: 'a;
}

impl Iterator for () {
    type Item<'a> = &'a ();
}

fn main() {}
