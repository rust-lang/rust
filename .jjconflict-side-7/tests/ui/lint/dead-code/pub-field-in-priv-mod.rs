#![deny(dead_code)]

fn main() {
    let _ = foo::S{f: false};
}

mod foo {
    pub struct S {
        pub f: bool, //~ ERROR field `f` is never read
    }
}
