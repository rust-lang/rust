// run-pass
// Tests multiple free variables being passed by value into an unboxed
// once closure as an optimization by codegen.  This used to hit an
// incorrect assert.

fn main() {
    let x = 2u8;
    let y = 3u8;
    assert_eq!((move || x + y)(), 5);
}
