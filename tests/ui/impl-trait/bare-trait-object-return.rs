//! Regression test for <https://github.com/rust-lang/rust/issues/18107>.

pub trait AbstractRenderer {}

fn _create_render(_: &()) ->
    dyn AbstractRenderer
//~^ ERROR return type cannot be a trait object without pointer indirection
{
    match 0 {
        _ => unimplemented!()
    }
}

fn main() {
}
