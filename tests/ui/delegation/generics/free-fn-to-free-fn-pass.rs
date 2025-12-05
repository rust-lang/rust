//@ run-pass
#![feature(fn_delegation)]
#![allow(incomplete_features)]

mod to_reuse {
    pub fn types<T, U>(x: U, y: T) -> (T, U) {
        (y, x)
    }
    pub fn late<'a, 'b>(x: &'a u8, y: &'b u8) -> u8 {
        *x + *y
    }
    pub fn early<'a: 'a>(x: &'a str) -> &'a str {
        x
    }
}

reuse to_reuse::types;
reuse to_reuse::late;
reuse to_reuse::early;

fn main() {
    assert_eq!(types(0, "str"), ("str", 0));
    assert_eq!(late(&1u8, &2u8), 3);
    {
        let s: &'static str = "hello world";
        assert_eq!(early::<'static>(s), "hello world");
    }
}
