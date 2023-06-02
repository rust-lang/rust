// This is a regression test for a `delay_span_bug` during interning when a constant
// evaluates to a (non-dangling) raw pointer.  For now this errors; potentially it
// could also be allowed.

const CONST_RAW: *const Vec<i32> = &Vec::new() as *const _;
//~^ ERROR unsupported untyped pointer in constant

fn main() {}
