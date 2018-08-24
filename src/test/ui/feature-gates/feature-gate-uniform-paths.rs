pub mod foo {
    pub use bar::Bar;
    //~^ ERROR unresolved import `bar`

    pub mod bar {
        pub struct Bar;
    }
}

fn main() {
    let _ = foo::Bar;
}
