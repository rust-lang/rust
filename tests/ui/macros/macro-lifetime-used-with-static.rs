// run-pass
macro_rules! foo {
    ($l:lifetime) => {
        fn f(arg: &$l str) -> &$l str {
            arg
        }
    }
}

pub fn main() {
    foo!('static);
    let x: &'static str = f("hi");
    assert_eq!("hi", x);
}
