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
fn num_cpus() {
    assert_eq!(num_cpus::get(), 1);
}


// FIXME: Remove this `cfg` once we fix https://github.com/rust-lang/miri/issues/1059.
// We cfg-gate the `should_panic` attribute and the `panic!` itself, so that the test
// stdout does not depend on the target.
#[test]
#[cfg_attr(not(windows), should_panic(expected="Explicit panic"))]
fn do_panic() { // In large, friendly letters :)
    #[cfg(not(windows))]
    panic!("Explicit panic from test!");
}

// FIXME: see above
#[test]
#[allow(unconditional_panic)]
#[cfg_attr(not(windows), should_panic(expected="the len is 0 but the index is 42"))]
fn fail_index_check() {
    #[cfg(not(windows))]
    [][42]
}
