//@ dont-require-annotations: NOTE

fn f() -> bool {
    assert!(1 < 2)
    //~^ ERROR mismatched types [E0308]
}

fn g() -> i32 {
    assert_eq!(1, 1)
    //~^ ERROR mismatched types [E0308]
}

fn h() -> bool {
    assert_ne!(1, 2)
    //~^ ERROR mismatched types [E0308]
}

// Test nested macros
macro_rules! g {
    () => {
        f!()
    };
}
macro_rules! f {
    () => {
        assert!(1 < 2)
        //~^ ERROR mismatched types [E0308]
    };
}
fn nested() -> bool {
    g!()
}

fn main() {}
