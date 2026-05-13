//@ run-pass
#![allow(unused_variables)]
// check that we don't accidentally capture upvars just because their name
// occurs in a path

fn assert_static<T: 'static>(_t: T) {}

mod foo {
    pub fn scope() {}
}

fn main() {
    let scope = &mut 0;
    assert_static(|| {
       foo::scope();
    });
}
