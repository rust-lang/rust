#[test]
fn bar() {
    assert_eq!(4, 4);
}

// Having more than 1 test does seem to make a difference
// (i.e., this calls ptr::swap which having just one test does not).
#[test]
fn baz() {
    assert_eq!(5, 5);
}

// A test that won't work on miri
#[cfg(not(feature = "cargo-miri"))]
#[test]
fn does_not_work_on_miri() {
    let x = 0u8;
    assert!(&x as *const _ as usize % 4 < 4);
}
