#[doc = "Operations and constants for `i16`"];

const min_value: i16 = -1i16 << 15i16;
const max_value: i16 = (-1i16 << 15i16) - 1i16;

pure fn min(x: i16, y: i16) -> i16 { if x < y { x } else { y } }
pure fn max(x: i16, y: i16) -> i16 { if x > y { x } else { y } }

pure fn add(x: i16, y: i16) -> i16 { x + y }
pure fn sub(x: i16, y: i16) -> i16 { x - y }
pure fn mul(x: i16, y: i16) -> i16 { x * y }
pure fn div(x: i16, y: i16) -> i16 { x / y }
pure fn rem(x: i16, y: i16) -> i16 { x % y }

pure fn lt(x: i16, y: i16) -> bool { x < y }
pure fn le(x: i16, y: i16) -> bool { x <= y }
pure fn eq(x: i16, y: i16) -> bool { x == y }
pure fn ne(x: i16, y: i16) -> bool { x != y }
pure fn ge(x: i16, y: i16) -> bool { x >= y }
pure fn gt(x: i16, y: i16) -> bool { x > y }

pure fn positive(x: i16) -> bool { x > 0i16 }
pure fn negative(x: i16) -> bool { x < 0i16 }
pure fn nonpositive(x: i16) -> bool { x <= 0i16 }
pure fn nonnegative(x: i16) -> bool { x >= 0i16 }

#[doc = "Iterate over the range [`lo`..`hi`)"]
fn range(lo: i16, hi: i16, it: fn(i16)) {
    let mut i = lo;
    while i < hi { it(i); i += 1i16; }
}

#[doc = "Computes the bitwise complement"]
pure fn compl(i: i16) -> i16 {
    u16::compl(i as u16) as i16
}

#[doc = "Computes the absolute value"]
pure fn abs(i: i16) -> i16 {
    if negative(i) { -i } else { i }
}
