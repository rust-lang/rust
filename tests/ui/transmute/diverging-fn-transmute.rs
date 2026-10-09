//! Regression test for issue #35849: transmute with panic in diverging function
//@ compile-flags: -Znext-solver=globally
fn assert_sizeof() -> ! {
    unsafe {
        ::std::mem::transmute::<f64, [u8; 8]>(panic!())
        //~^ ERROR mismatched types
    }
}

fn main() {}
