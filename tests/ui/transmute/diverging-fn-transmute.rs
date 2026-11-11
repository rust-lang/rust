//! Regression test for issue #35849: transmute with panic in diverging function
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
fn assert_sizeof() -> ! {
    unsafe {
        ::std::mem::transmute::<f64, [u8; 8]>(panic!())
        //~^ ERROR mismatched types
    }
}

fn main() {}
