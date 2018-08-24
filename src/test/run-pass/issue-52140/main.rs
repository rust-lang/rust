// aux-build:some_crate.rs
// edition:2018

mod foo {
    pub use some_crate;
}

fn main() {
    ::some_crate::hello();
    foo::some_crate::hello();
}
