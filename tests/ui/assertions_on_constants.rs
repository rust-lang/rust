//FIXME: suggestions are wrongly expanded, this should be fixed along with #7843
#![allow(non_fmt_panics)]

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

    // Don't lint on this:
    assert!(cfg!(feature = "hey") || cfg!(not(feature = "asdf")));
}
