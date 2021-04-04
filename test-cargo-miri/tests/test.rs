use rand::{SeedableRng, Rng, rngs::SmallRng};

// Having more than 1 test does seem to make a difference
// (i.e., this calls ptr::swap which having just one test does not).
#[test]
fn simple1() {
    assert_eq!(4, 4);
}

#[test]
fn simple2() {
    assert_ne!(42, 24);
}

// A test that won't work on miri (tests disabling tests).
#[test]
#[cfg_attr(miri, ignore)]
fn does_not_work_on_miri() {
    let x = 0u8;
    assert!(&x as *const _ as usize % 4 < 4);
}

// We also use this to test some external crates, that we cannot depend on in the compiletest suite.

#[test]
fn entropy_rng() {
    // Test `getrandom` directly (in multiple different versions).
    let mut data = vec![0; 16];
    getrandom_1::getrandom(&mut data).unwrap();
    getrandom_2::getrandom(&mut data).unwrap();

    // Try seeding with "real" entropy.
    let mut rng = SmallRng::from_entropy();
    let _val = rng.gen::<i32>();
    let _val = rng.gen::<isize>();
    let _val = rng.gen::<i128>();

    // Also try per-thread RNG.
    let mut rng = rand::thread_rng();
    let _val = rng.gen::<i32>();
    let _val = rng.gen::<isize>();
    let _val = rng.gen::<i128>();
}

#[test]
fn cargo_env() {
    assert_eq!(env!("CARGO_PKG_NAME"), "cargo-miri-test");
    env!("CARGO_BIN_EXE_cargo-miri-test"); // Asserts that this exists.
}

#[test]
#[should_panic(expected="Explicit panic")]
fn do_panic() { // In large, friendly letters :)
    panic!("Explicit panic from test!");
}

#[test]
#[allow(unconditional_panic)]
#[should_panic(expected="the len is 0 but the index is 42")]
fn fail_index_check() {
    [][42]
}
