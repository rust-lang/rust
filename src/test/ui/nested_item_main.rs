// run-pass
// aux-build:nested_item.rs


extern crate nested_item;

pub fn main() {
    assert_eq!(2, nested_item::foo::<()>());
    assert_eq!(2, nested_item::foo::<isize>());
}
