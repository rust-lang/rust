// run-pass

#![feature(const_fn_mut_refs)]

struct Foo {
    x: i32
}

const fn bar(foo: &mut Foo) -> i32 {
    foo.x + 1
}

fn main() {
    assert_eq!(bar(&mut Foo{x: 0}), 1);
}
