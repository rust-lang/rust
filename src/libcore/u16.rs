#[doc = "Operations and constants for `u16`"];

const min_value: u16 = 0u16;
const max_value: u16 = 0u16 - 1u16;

pure fn min(x: u16, y: u16) -> u16 { if x < y { x } else { y } }
pure fn max(x: u16, y: u16) -> u16 { if x > y { x } else { y } }

pure fn add(x: u16, y: u16) -> u16 { x + y }
pure fn sub(x: u16, y: u16) -> u16 { x - y }
pure fn mul(x: u16, y: u16) -> u16 { x * y }
pure fn div(x: u16, y: u16) -> u16 { x / y }
pure fn rem(x: u16, y: u16) -> u16 { x % y }

pure fn lt(x: u16, y: u16) -> bool { x < y }
pure fn le(x: u16, y: u16) -> bool { x <= y }
pure fn eq(x: u16, y: u16) -> bool { x == y }
pure fn ne(x: u16, y: u16) -> bool { x != y }
pure fn ge(x: u16, y: u16) -> bool { x >= y }
pure fn gt(x: u16, y: u16) -> bool { x > y }

pure fn is_positive(x: u16) -> bool { x > 0u16 }
pure fn is_negative(x: u16) -> bool { x < 0u16 }
pure fn is_nonpositive(x: u16) -> bool { x <= 0u16 }
pure fn is_nonnegative(x: u16) -> bool { x >= 0u16 }

#[doc = "Iterate over the range [`lo`..`hi`)"]
fn range(lo: u16, hi: u16, it: fn(u16)) {
    let mut i = lo;
    while i < hi { it(i); i += 1u16; }
}

#[doc = "Computes the bitwise complement"]
pure fn compl(i: u16) -> u16 {
    max_value ^ i
}
