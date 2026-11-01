//@ run-pass
fn foo(x: &usize) -> &usize { x }
fn bar(x: &usize) -> usize { *x }

pub fn main() {
    let p: Box<_> = Box::new(3);
    assert_eq!(bar(foo(&*p)), 3);
}
