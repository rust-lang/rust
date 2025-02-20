#![allow(non_fmt_panics, clippy::needless_bool, clippy::eq_op)]

macro_rules! assert_const {
    ($len:expr) => {
        assert!($len > 0);
        debug_assert!($len < 0);
    };
}
fn main() {
    assert!(true);
    //~^ assertions_on_constants

    assert!(false);
    //~^ assertions_on_constants

    assert!(true, "true message");
    //~^ assertions_on_constants

    assert!(false, "false message");
    //~^ assertions_on_constants

    let msg = "panic message";
    assert!(false, "{}", msg.to_uppercase());
    //~^ assertions_on_constants

    const B: bool = true;
    assert!(B);
    //~^ assertions_on_constants

    const C: bool = false;
    assert!(C);
    //~^ assertions_on_constants

    assert!(C, "C message");
    //~^ assertions_on_constants

    debug_assert!(true);
    //~^ assertions_on_constants

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

    const _: () = assert!(true);
    //~^ assertions_on_constants

    assert!(8 == (7 + 1));
    //~^ assertions_on_constants

    // Don't lint if the value is dependent on a defined constant:
    const N: usize = 1024;
    const _: () = assert!(N.is_power_of_two());
}

const _: () = {
    assert!(true);
    //~^ assertions_on_constants

    assert!(8 == (7 + 1));
};
