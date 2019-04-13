// compile-pass
// aux-build:variants.rs

extern crate variants;

const S: u8 = 0;
use variants::NonExhaustiveVariants::Struct as S;

fn main() {}
