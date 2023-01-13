// build-pass (FIXME(62277): could be check-pass?)
// aux-build:variants.rs

extern crate variants;

const S: u8 = 0;

// OK, `Struct` in value namespace is crate-private, so it's filtered away
// and there's no conflict with the previously defined `const S`.
use variants::NonExhaustiveVariants::Struct as S;

fn main() {}
