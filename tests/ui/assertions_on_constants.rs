#![allow(non_fmt_panics, clippy::needless_bool)]

macro_rules! assert_const {
    ($len:expr) => {
        assert!($len > 0);
        debug_assert!($len < 0);
    };
}
fn main() {
    assert!(true);
    assert!(false);
    assert!(true, "true message");
    assert!(false, "false message");

    let msg = "panic message";
    assert!(false, "{}", msg.to_uppercase());

    const B: bool = true;
    assert!(B);

    const C: bool = false;
    assert!(C);
    assert!(C, "C message");

    debug_assert!(true);
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
