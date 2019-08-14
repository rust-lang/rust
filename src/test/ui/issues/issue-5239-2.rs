// run-pass
// Regression test for issue #5239


pub fn main() {
    let _f = |ref x: isize| { *x };
    let foo = 10;
    assert_eq!(_f(foo), 10);
}
