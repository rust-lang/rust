//@ run-pass
#![allow(dead_code)]
// Make sure const bounds work on things, and test that a few types
// are const.


fn foo<T: Sync>(x: T) -> T { x }

struct F { field: isize }

pub fn main() {
    /*foo(1);
    foo("hi".to_string());
    foo(vec![1, 2, 3]);
    foo(F{field: 42});
    foo((1, 2));
    foo(@1);*/
    foo(Box::new(1));
}
