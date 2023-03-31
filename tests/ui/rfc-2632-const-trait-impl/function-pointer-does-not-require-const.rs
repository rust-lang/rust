// check-pass

type I32Cmp = fn(&i32, &i32) -> core::cmp::Ordering;
pub const fn min_by_i32() -> fn(i32, i32, I32Cmp) -> i32 {
    core::cmp::min_by
}

fn main() {}
