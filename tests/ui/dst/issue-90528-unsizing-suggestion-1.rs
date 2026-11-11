// Issue #90528: provide helpful suggestions when a trait bound is unsatisfied
// due to a missed unsizing coercion.
//
// This test exercises array literals and a trait implemented on immutable slices.

trait Read {}

impl Read for &[u8] {}

fn wants_read(_: impl Read) {}

fn main() {
    wants_read([0u8]);
    //~^ ERROR the trait bound `[u8; 1]: Read` is not satisfied
    wants_read(&[0u8]);
    //~^ ERROR the trait bound `&[u8; 1]: Read` is not satisfied
    wants_read(&[0u8][..]);
    wants_read(&mut [0u8]);
    //~^ ERROR the trait bound `&mut [u8; 1]: Read` is not satisfied
}
