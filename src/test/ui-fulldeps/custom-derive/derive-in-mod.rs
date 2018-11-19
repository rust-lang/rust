// compile-pass
// aux-build:plugin.rs

extern crate plugin;

mod inner {
    use plugin::WithHelper;

    #[derive(WithHelper)]
    struct S;
}

fn main() {}
