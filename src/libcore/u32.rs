#[doc = "Operations and constants for `u32`"];

const min_value: u32 = 0u32;
const max_value: u32 = 0u32 - 1u32;

pure fn min(x: u32, y: u32) -> u32 { if x < y { x } else { y } }
pure fn max(x: u32, y: u32) -> u32 { if x > y { x } else { y } }

pure fn add(x: u32, y: u32) -> u32 { ret x + y; }
pure fn sub(x: u32, y: u32) -> u32 { ret x - y; }
pure fn mul(x: u32, y: u32) -> u32 { ret x * y; }
pure fn div(x: u32, y: u32) -> u32 { ret x / y; }
pure fn rem(x: u32, y: u32) -> u32 { ret x % y; }

pure fn lt(x: u32, y: u32) -> bool { ret x < y; }
pure fn le(x: u32, y: u32) -> bool { ret x <= y; }
pure fn eq(x: u32, y: u32) -> bool { ret x == y; }
pure fn ne(x: u32, y: u32) -> bool { ret x != y; }
pure fn ge(x: u32, y: u32) -> bool { ret x >= y; }
pure fn gt(x: u32, y: u32) -> bool { ret x > y; }

#[doc = "Iterate over the range [`lo`..`hi`)"]
fn range(lo: u32, hi: u32, it: fn(u32)) {
    let mut i = lo;
    while i < hi { it(i); i += 1u32; }
}

#[doc = "Computes the bitwise complement"]
pure fn compl(i: u32) -> u32 {
    max_value ^ i
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
