mod a {
    pub mod b {
        pub mod c {
            pub struct S<T>(T);
        }
    }
}

macro_rules! import {
    ($p: path) => (use $p;);
}

fn f1() {
    import! { a::b::c::S<u8> } //~ ERROR unexpected generic arguments in path
}
fn f2() {
    import! { a::b::c::S<> } //~ ERROR unexpected generic arguments in path
}

fn main() {}
