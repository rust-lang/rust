//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/2723
pub unsafe fn f(xs: Vec<isize> ) {
    xs.iter().map(|_x| { unsafe fn q() { panic!(); } }).collect::<Vec<()>>();
}
