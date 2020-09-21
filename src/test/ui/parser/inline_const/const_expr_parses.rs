// check-pass
// compile-flags: -Z parse-only

#![feature(inline_const)]
fn foo() -> i32 {
    const {
        let x = 5 + 10;
        x / 3
    }
}
