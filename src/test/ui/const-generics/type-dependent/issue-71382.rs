// run-pass
#![feature(const_generics)]
#![allow(incomplete_features)]
#![feature(const_compare_raw_pointers)]

struct Test;

fn pass() -> u8 {
    42
}

impl Test {
    pub fn call_me(&self) -> u8 {
        self.test::<pass>()
    }

    fn test<const FN: fn() -> u8>(&self) -> u8 {
        FN()
    }
}

fn main() {
    let x = Test;
    assert_eq!(x.call_me(), 42);
}
