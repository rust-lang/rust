// run-pass
#![feature(box_syntax)]

fn foo(x: &usize) -> &usize { x }
fn bar(x: &usize) -> usize { *x }

pub fn main() {
    let p: Box<_> = box 3;
    assert_eq!(bar(foo(&*p)), 3);
}
