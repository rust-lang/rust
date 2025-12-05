//! Issue: <https://github.com/rust-lang/rust/issues/131227>
//! Test that constant propagation in SwitchInt does not crash
//! when encountering a ptr-to-int transmute.

//@ check-pass
//@ compile-flags: -Zmir-enable-passes=+InstSimplify-before-inline,+DataflowConstProp

#![crate_type = "lib"]

static mut G: i32 = 0;

pub fn myfunc() -> i32 {
    let var = &raw mut G;
    let u: usize = unsafe { std::mem::transmute(var) };
    match u {
        0 => 0,
        _ => 1,
    }
}
