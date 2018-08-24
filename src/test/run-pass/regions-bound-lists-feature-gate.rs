#![feature(issue_5723_bootstrap)]

trait Foo {
    fn dummy(&self) { }
}

fn foo<'a>(x: Box<Foo + 'a>) {
}

fn bar<'a, T: 'a>() {
}

pub fn main() { }
