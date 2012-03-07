#[doc = "Operations and constants for `i32`"];

const min_value: i32 = -1i32 << 31i32;
const max_value: i32 = (-1i32 << 31i32) - 1i32;

pure fn min(x: i32, y: i32) -> i32 { if x < y { x } else { y } }
pure fn max(x: i32, y: i32) -> i32 { if x > y { x } else { y } }

pure fn add(x: i32, y: i32) -> i32 { x + y }
pure fn sub(x: i32, y: i32) -> i32 { x - y }
pure fn mul(x: i32, y: i32) -> i32 { x * y }
pure fn div(x: i32, y: i32) -> i32 { x / y }
pure fn rem(x: i32, y: i32) -> i32 { x % y }

pure fn lt(x: i32, y: i32) -> bool { x < y }
pure fn le(x: i32, y: i32) -> bool { x <= y }
pure fn eq(x: i32, y: i32) -> bool { x == y }
pure fn ne(x: i32, y: i32) -> bool { x != y }
pure fn ge(x: i32, y: i32) -> bool { x >= y }
pure fn gt(x: i32, y: i32) -> bool { x > y }

pure fn positive(x: i32) -> bool { x > 0i32 }
pure fn negative(x: i32) -> bool { x < 0i32 }
pure fn nonpositive(x: i32) -> bool { x <= 0i32 }
pure fn nonnegative(x: i32) -> bool { x >= 0i32 }

#[doc = "Iterate over the range [`lo`..`hi`)"]
fn range(lo: i32, hi: i32, it: fn(i32)) {
    let mut i = lo;
    while i < hi { it(i); i += 1i32; }
}

#[doc = "Computes the bitwise complement"]
fn compl(i: i32) -> i32 {
    u32::compl(i as u32) as i32
}

#[doc = "Computes the absolute value"]
fn abs(i: i32) -> i32 {
    if negative(i) { -i } else { i }
}
