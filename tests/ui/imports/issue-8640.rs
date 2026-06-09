#[allow(unused_imports)]

mod foo {
    use crate::baz::bar;
    mod bar {}
    //~^ ERROR the name `bar` is defined multiple times
}
mod baz { pub mod bar {} }

fn main() {}
