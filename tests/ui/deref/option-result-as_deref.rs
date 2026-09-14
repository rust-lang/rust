//! Regression test for <https://github.com/rust-lang/rust/issues/50264>.

fn main() {
    let _m = &Some(42).as_deref();
//~^ ERROR the method
    let _e = &mut Some(42).as_deref_mut();
//~^ ERROR the method
    let _o = &Ok(42).as_deref();
//~^ ERROR the method
    let _w = &mut Ok(42).as_deref_mut();
//~^ ERROR the method
}
