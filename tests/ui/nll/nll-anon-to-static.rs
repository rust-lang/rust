//! regression test for issue <https://github.com/rust-lang/rust/issues/46983>
fn foo(x: &u32) -> &'static u32 {
    &*x
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
