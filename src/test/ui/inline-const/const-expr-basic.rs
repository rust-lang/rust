// run-pass

#![allow(incomplete_features)]
#![feature(inline_const)]
fn foo() -> u32 {
    const {
        let x = 5u32 + 10;
        x / 3
    }
}

fn main() {
    assert_eq!(5, foo());
}
