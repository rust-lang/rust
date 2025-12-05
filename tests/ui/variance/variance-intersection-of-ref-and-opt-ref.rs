//@ run-pass
// Elaborated version of the opening example from RFC 738. This failed
// to compile before variance because invariance of `Option` prevented
// us from approximating the lifetimes of `field1` and `field2` to a
// common intersection.

#![allow(dead_code)]

struct List<'l> {
    field1: &'l i32,
    field2: Option<&'l i32>,
}

fn foo(field1: &i32, field2: Option<&i32>) -> i32 {
    let list = List { field1: field1, field2: field2 };
    *list.field1 + list.field2.cloned().unwrap_or(0)
}

fn main() {
    let x = 22;
    let y = Some(3);
    let z = None;
    assert_eq!(foo(&x, y.as_ref()), 25);
    assert_eq!(foo(&x, z.as_ref()), 22);
}
