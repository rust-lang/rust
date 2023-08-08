// provide custom privacy error for sealed traits
pub mod a {
    pub trait Sealed: self::b::Hidden {
        fn foo() {}
    }

    struct X;
    impl Sealed for X {}
    impl self::b::Hidden for X {}

    mod b {
        pub trait Hidden {}
    }
}

struct S;
impl a::Sealed for S {} //~ ERROR the trait bound `S: Hidden` is not satisfied

fn main() {}
