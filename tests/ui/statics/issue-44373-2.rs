//@ check-pass
#![allow(dead_code)]

struct Foo(bool);

struct Container(&'static [&'static Foo]);

static FOO: Foo = Foo(true);
static CONTAINER: Container = Container(&[&FOO]);

fn main() {}
