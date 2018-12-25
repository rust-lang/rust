// run-pass
#![allow(dead_code)]
#![allow(stable_features)]

#![feature(issue_5723_bootstrap)]

trait Foo {
    fn dummy(&self) { }
}

fn foo<'a, 'b, 'c:'a+'b, 'd>() {
}

pub fn main() { }
