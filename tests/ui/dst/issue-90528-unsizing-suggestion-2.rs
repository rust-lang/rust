// Issue #90528: provide helpful suggestions when a trait bound is unsatisfied
// due to a missed unsizing coercion.
//
// This test exercises array variables and a trait implemented on immmutable slices.

trait Read {}

impl Read for &[u8] {}

fn wants_read(_: impl Read) {}

fn main() {
    let x = [0u8];
    wants_read(x);
    //~^ ERROR trait `Read` is not implemented for `[u8; 1]`
    wants_read(&x);
    //~^ ERROR trait `Read` is not implemented for `&[u8; 1]`
    wants_read(&x[..]);

    let x = &[0u8];
    wants_read(x);
    //~^ ERROR trait `Read` is not implemented for `&[u8; 1]`
    wants_read(&x);
    //~^ ERROR trait `Read` is not implemented for `&&[u8; 1]`
    wants_read(*x);
    //~^ ERROR trait `Read` is not implemented for `[u8; 1]`
    wants_read(&x[..]);
}
