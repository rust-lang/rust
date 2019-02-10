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

    const B: bool = true;
    assert!(B);

    const C: bool = false;
    assert!(C);

    debug_assert!(true);
    assert_const!(3);
    assert_const!(-1);
}
