/// Check that only `&X: Debug` is required, not `X: Debug`
//@check-pass

use std::fmt::Debug;
use std::fmt::Formatter;

struct X;

impl Debug for &X {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.write_str("X")
    }
}

fn main() {
    dbg!(X);
}
