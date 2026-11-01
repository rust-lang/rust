// https://github.com/rust-lang/rust/issues/7012
//@ run-pass
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

/*
# Comparison of static arrays

The expected behaviour would be that `test == test1`, therefore 'true'
would be printed, however the below prints false.
*/

struct signature<'a> { pattern : &'a [u32] }

static test1: signature<'static> =  signature {
  pattern: &[0x243f6a88,0x85a308d3,0x13198a2e,0x03707344,0xa4093822,0x299f31d0]
};

pub fn main() {
  let test: &[u32] = &[0x243f6a88,0x85a308d3,0x13198a2e,
                       0x03707344,0xa4093822,0x299f31d0];
  println!("{}",test==test1.pattern);
}
