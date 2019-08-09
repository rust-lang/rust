pub trait AbstractRenderer {}

fn _create_render(_: &()) ->
    dyn AbstractRenderer
//~^ ERROR the size for values of type
{
    match 0 {
        _ => unimplemented!()
    }
}

fn main() {
}
