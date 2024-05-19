extern crate foo1;
extern crate foo2;

fn main() {
    let a = foo1::foo();
    let b = foo2::foo();
    assert!(a as *const _ != b as *const _);
}
