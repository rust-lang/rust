// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(stable_features)]

#![feature(issue_5723_bootstrap)]

trait Foo {
    fn dummy(&self) { }
}

fn foo<'a>(x: Box<dyn Foo + 'a>) {
}

fn bar<'a, T: 'a>() {
}

pub fn main() { }
