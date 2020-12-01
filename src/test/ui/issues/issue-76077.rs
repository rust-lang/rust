pub mod foo {
    pub struct Foo {
        you_cant_use_this_field: bool,
    }
}

fn main() {
    foo::Foo {};
    //~^ ERROR cannot construct `Foo` with struct literal syntax due to inaccessible fields
}
