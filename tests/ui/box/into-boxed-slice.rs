//@ run-pass
#![feature(box_into_boxed_slice)]
fn main() {
    assert_eq!(Box::into_boxed_slice(Box::new(5u8)), Box::new([5u8]) as Box<[u8]>);
    assert_eq!(Box::into_boxed_slice(Box::new([25u8])), Box::new([[25u8]]) as Box<[[u8; 1]]>);
    let a: Box<[Box<[u8; 1]>]> = Box::into_boxed_slice(Box::new(Box::new([5u8])));
    let b: Box<[Box<[u8; 1]>]> = Box::new([Box::new([5u8])]);
    assert_eq!(a, b);
}
