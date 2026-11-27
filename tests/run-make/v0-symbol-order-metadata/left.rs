pub struct Left;

#[inline(never)]
pub fn left_function<T>(x: T) -> T {
    std::hint::black_box(x)
}
