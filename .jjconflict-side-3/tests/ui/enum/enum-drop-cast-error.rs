//! Check that trying to cast an enum with a Drop impl to an integer is rejected.
//!
//! Issue: <https://github.com/rust-lang/rust/issues/35941>

enum E {
    A = 0,
}

impl Drop for E {
    fn drop(&mut self) {
        println!("Drop");
    }
}

fn main() {
    let e = E::A;
    let i = e as u32;
    //~^ ERROR cannot cast enum `E` into integer `u32` because it implements `Drop`
}
