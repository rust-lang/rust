//@ run-pass

#![allow(non_camel_case_types)]

trait get {
    fn get(self) -> isize;
}

// Note: impl on a slice; we're checking that the pointers below
// correctly get borrowed to `&`. (similar to impling for `isize`, with
// `&self` instead of `self`.)
impl<'a> get for &'a isize {
    fn get(self) -> isize {
        return *self;
    }
}

pub fn main() {
    let x: Box<_> = 6.into();
    let y = x.get();
    println!("y={}", y);
    assert_eq!(y, 6);

    let x = &6;
    let y = x.get();
    println!("y={}", y);
    assert_eq!(y, 6);
}
