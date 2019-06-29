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

// A test that won't work on miri (tests disabling tests)
#[cfg(not(miri))]
#[test]
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
