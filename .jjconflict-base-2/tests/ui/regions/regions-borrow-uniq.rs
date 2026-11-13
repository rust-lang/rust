//@ run-pass

fn foo(x: &usize) -> usize {
    *x
}

pub fn main() {
    let p: Box<_> = Box::new(3);
    let r = foo(&*p);
    assert_eq!(r, 3);
}
