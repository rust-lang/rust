// ignore-tidy-dbg

/// Test for <https://github.com/rust-lang/rust/issues/153850>:
/// `dbg!` shouldn't drop arguments' temporaries.
#[test]
fn no_dropping_temps() {
    fn temp() {}

    *dbg!(&temp());
    *dbg!(&temp(), 1).0;
    *dbg!(0, &temp()).1;
    *dbg!(0, &temp(), 2).1;
}
