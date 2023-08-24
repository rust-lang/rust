#![allow(non_fmt_panics, clippy::needless_bool)]

macro_rules! assert_const {
    ($len:expr) => {
        assert!($len > 0);
        debug_assert!($len < 0);
    };
}
fn main() {
    assert!(true);
    //~^ ERROR: `assert!(true)` will be optimized out by the compiler
    assert!(false);
    //~^ ERROR: `assert!(false)` should probably be replaced
    assert!(true, "true message");
    //~^ ERROR: `assert!(true)` will be optimized out by the compiler
    assert!(false, "false message");
    //~^ ERROR: `assert!(false, ..)` should probably be replaced

    let msg = "panic message";
    assert!(false, "{}", msg.to_uppercase());
    //~^ ERROR: `assert!(false, ..)` should probably be replaced

    const B: bool = true;
    assert!(B);
    //~^ ERROR: `assert!(true)` will be optimized out by the compiler

    const C: bool = false;
    assert!(C);
    //~^ ERROR: `assert!(false)` should probably be replaced
    assert!(C, "C message");
    //~^ ERROR: `assert!(false, ..)` should probably be replaced

    debug_assert!(true);
    //~^ ERROR: `debug_assert!(true)` will be optimized out by the compiler
    // Don't lint this, since there is no better way for expressing "Only panic in debug mode".
    debug_assert!(false); // #3948
    assert_const!(3);
    assert_const!(-1);

    // Don't lint if based on `cfg!(..)`:
    assert!(cfg!(feature = "hey") || cfg!(not(feature = "asdf")));

    let flag: bool = cfg!(not(feature = "asdf"));
    assert!(flag);

    const CFG_FLAG: &bool = &cfg!(feature = "hey");
    assert!(!CFG_FLAG);
}
