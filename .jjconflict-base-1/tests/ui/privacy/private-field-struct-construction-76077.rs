// https://github.com/rust-lang/rust/issues/76077
pub mod foo {
    pub struct Foo {
        you_cant_use_this_field: bool,
    }
}

fn main() {
    foo::Foo {};
    //~^ ERROR cannot construct `Foo` with struct literal syntax due to private fields
}
