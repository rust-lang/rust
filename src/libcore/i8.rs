#[doc = "Operations and constants for `i8`"];

const min_value: i8 = -1i8 << 7i8;
const max_value: i8 = (-1i8 << 7i8) - 1i8;

pure fn min(x: i8, y: i8) -> i8 { if x < y { x } else { y } }
pure fn max(x: i8, y: i8) -> i8 { if x > y { x } else { y } }

pure fn add(x: i8, y: i8) -> i8 { x + y }
pure fn sub(x: i8, y: i8) -> i8 { x - y }
pure fn mul(x: i8, y: i8) -> i8 { x * y }
pure fn div(x: i8, y: i8) -> i8 { x / y }
pure fn rem(x: i8, y: i8) -> i8 { x % y }

pure fn lt(x: i8, y: i8) -> bool { x < y }
pure fn le(x: i8, y: i8) -> bool { x <= y }
pure fn eq(x: i8, y: i8) -> bool { x == y }
pure fn ne(x: i8, y: i8) -> bool { x != y }
pure fn ge(x: i8, y: i8) -> bool { x >= y }
pure fn gt(x: i8, y: i8) -> bool { x > y }

pure fn positive(x: i8) -> bool { x > 0i8 }
pure fn negative(x: i8) -> bool { x < 0i8 }
pure fn nonpositive(x: i8) -> bool { x <= 0i8 }
pure fn nonnegative(x: i8) -> bool { x >= 0i8 }

#[doc = "Iterate over the range [`lo`..`hi`)"]
fn range(lo: i8, hi: i8, it: fn(i8)) {
    let mut i = lo;
    while i < hi { it(i); i += 1i8; }
}

#[doc = "Computes the bitwise complement"]
pure fn compl(i: i8) -> i8 {
    u8::compl(i as u8) as i8
}

#[doc = "Computes the absolute value"]
pure fn abs(i: i8) -> i8 {
    if negative(i) { -i } else { i }
}
