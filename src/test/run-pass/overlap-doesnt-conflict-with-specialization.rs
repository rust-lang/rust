#![feature(overlapping_marker_traits)]
#![feature(specialization)]

trait MyMarker {}

impl<T> MyMarker for T {}
impl<T> MyMarker for Vec<T> {}

fn foo<T: MyMarker>(t: T) -> T {
    t
}

fn main() {
    assert_eq!(1, foo(1));
    assert_eq!(2.0, foo(2.0));
    assert_eq!(vec![1], foo(vec![1]));
}
