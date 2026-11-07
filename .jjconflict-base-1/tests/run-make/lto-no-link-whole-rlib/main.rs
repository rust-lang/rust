extern crate lib1;
extern crate lib2;

fn main() {
    assert_eq!(lib1::foo1(), 2);
    assert_eq!(lib2::foo2(), 2);
}
