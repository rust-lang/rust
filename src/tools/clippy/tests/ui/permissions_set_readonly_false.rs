#![allow(unused)]
#![warn(clippy::permissions_set_readonly_false)]

use std::fs::File;

struct A;

impl A {
    pub fn set_readonly(&mut self, b: bool) {}
}

fn set_readonly(b: bool) {}

fn main() {
    let f = File::create("foo.txt").unwrap();
    let metadata = f.metadata().unwrap();
    let mut permissions = metadata.permissions();
    // lint here
    permissions.set_readonly(false);
    //~^ permissions_set_readonly_false

    // no lint
    permissions.set_readonly(true);

    let mut a = A;
    // no lint here - a is not of type std::fs::Permissions
    a.set_readonly(false);

    // no lint here - plain function
    set_readonly(false);
}
