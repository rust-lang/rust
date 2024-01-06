#![feature(rustc_attrs)]

#[rustc_layout_scalar_valid_range_start(1)]
#[repr(transparent)]
pub(crate) struct NonZero<T>(pub(crate) T);
fn main() {
    let mut x = unsafe { NonZero(1) };
    let y = &mut x.0; //~ ERROR mutation of layout constrained field is unsafe
    if let Some(NonZero(ref mut y)) = Some(x) {} //~ ERROR mutation of layout constrained field is unsafe
}
