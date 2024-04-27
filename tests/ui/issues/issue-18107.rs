pub trait AbstractRenderer {}

fn _create_render(_: &()) ->
    dyn AbstractRenderer
//~^ ERROR return type cannot have an unboxed trait object
{
    match 0 {
        _ => unimplemented!()
    }
}

fn main() {
}
