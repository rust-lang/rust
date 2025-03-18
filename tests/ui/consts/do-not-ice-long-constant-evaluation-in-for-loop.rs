// The test confirms ICE-125323 is fixed.
//
// This warning makes sure the fix doesn't throw everything with errors to dead.
#![warn(unused)]
fn should_not_be_dead() {}

fn main() {
    for _ in 0..0 {
        [(); loop {}]; //~ ERROR constant evaluation is taking a long time
    }

    should_not_be_dead();
}
