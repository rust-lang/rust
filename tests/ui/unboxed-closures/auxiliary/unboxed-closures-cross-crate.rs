use std::ops::Add;

#[inline]
pub fn has_closures() -> usize {
    let x = 1;
    let mut f = move || x;
    let y = 1;
    let g = || y;
    f() + g()
}

pub fn has_generic_closures<T: Add<Output=T> + Copy>(x: T, y: T) -> T {
    let mut f = move || x;
    let g = || y;
    f() + g()
}
