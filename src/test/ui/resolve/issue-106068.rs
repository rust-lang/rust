pub mod foo {
    pub mod bar {
        pub struct Baz;
    }
}

fn main() {
    let _: bar::Baz = unimplemented!();
    //~^ ERROR failed to resolve: unresolved import [E0433]
}
