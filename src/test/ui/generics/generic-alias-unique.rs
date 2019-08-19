// run-pass
#![feature(box_syntax)]

fn id<T:Send>(t: T) -> T { return t; }

pub fn main() {
    let expected: Box<_> = box 100;
    let actual = id::<Box<isize>>(expected.clone());
    println!("{}", *actual);
    assert_eq!(*expected, *actual);
}
