pub struct Right;

#[inline(never)]
pub fn right_function<T>(x: T) -> T {
    std::hint::black_box(x)
}
