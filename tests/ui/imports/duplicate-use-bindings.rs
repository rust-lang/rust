//! Test that duplicate use bindings in same namespace produce error

mod foo {
    pub use self::bar::X;
    use self::bar::X;
    //~^ ERROR the name `X` is defined multiple times

    mod bar {
        pub struct X;
    }
}

fn main() {
    let _ = foo::X;
}
