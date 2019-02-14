extern crate rand;

use rand::{Rng, SeedableRng};

#[test]
fn simple() {
    assert_eq!(4, 4);
}

// Having more than 1 test does seem to make a difference
// (i.e., this calls ptr::swap which having just one test does not).
#[test]
fn rng() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xdeadcafe);
    let x: u32 = rng.gen();
    let y: u32 = rng.gen();
    assert_ne!(x, y);
}

// A test that won't work on miri
#[cfg(not(miri))]
#[test]
fn does_not_work_on_miri() {
    let x = 0u8;
    assert!(&x as *const _ as usize % 4 < 4);
}
