use rand::{FromEntropy, Rng, rngs::SmallRng};

// Having more than 1 test does seem to make a difference
// (i.e., this calls ptr::swap which having just one test does not).
#[test]
fn simple() {
    assert_eq!(4, 4);
}

// Having more than 1 test does seem to make a difference
// (i.e., this calls ptr::swap which having just one test does not).
#[test]
fn entropy_rng() {
    // Use this opportunity to test querying the RNG (needs an external crate, hence tested here and not in the compiletest suite)
    let mut rng = SmallRng::from_entropy();
    let _val = rng.gen::<i32>();

    // Also try per-thread RNG.
    let mut rng = rand::thread_rng();
    let _val = rng.gen::<i32>();
}

// A test that won't work on miri
#[cfg(not(miri))]
#[test]
fn does_not_work_on_miri() {
    let x = 0u8;
    assert!(&x as *const _ as usize % 4 < 4);
}
