#[test]
fn simple() {
    assert_eq!(4, 4);
}

// A test that won't work on miri (tests disabling tests).
#[test]
#[cfg_attr(miri, ignore)]
fn does_not_work_on_miri() {
    // Only do this where inline assembly is stable.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        std::arch::asm!("foo");
    }
}

// Make sure integration tests can access both dependencies and dev-dependencies
#[test]
fn deps() {
    {
        use byteorder::{BigEndian, ByteOrder};
        let _n = <BigEndian as ByteOrder>::read_u64(&[1, 2, 3, 4, 5, 6, 7, 8]);
    }
    {
        use byteorder_2::{BigEndian, ByteOrder};
        let _n = <BigEndian as ByteOrder>::read_u64(&[1, 2, 3, 4, 5, 6, 7, 8]);
    }
}

#[test]
fn cargo_env() {
    assert_eq!(env!("CARGO_PKG_NAME"), "cargo-miri-test");
    env!("CARGO_BIN_EXE_cargo-miri-test"); // Asserts that this exists.
}

#[test]
#[should_panic(expected = "Explicit panic")]
fn do_panic() // In large, friendly letters :)
{
    panic!("Explicit panic from test!");
}

// A different way of raising a panic
#[test]
#[allow(unconditional_panic)]
#[should_panic(expected = "the len is 0 but the index is 42")]
fn fail_index_check() {
    [][42]
}
