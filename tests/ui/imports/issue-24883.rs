// check-pass

mod a {
    pub mod b { pub struct Foo; }

    pub mod c {
        use super::b;
        pub struct Bar(pub b::Foo);
    }

    pub use self::c::*;
}

fn main() {
    let _ = a::c::Bar(a::b::Foo);
    let _ = a::Bar(a::b::Foo);
}
