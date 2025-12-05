//! Regression test for invalid suggestion for `&raw const expr` reported in
//! <https://github.com/rust-lang/rust/issues/127562>.

fn main() {
    let val = 2;
    let ptr = &raw const val;
    unsafe { *ptr = 3; } //~ ERROR cannot assign to `*ptr`, which is behind a `*const` pointer
}
