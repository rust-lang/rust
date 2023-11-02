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

pub mod c {
    pub trait Sealed: self::d::Hidden {
        fn foo() {}
    }

    struct X;
    impl Sealed for X {}
    impl self::d::Hidden for X {}

    struct Y;
    impl Sealed for Y {}
    impl self::d::Hidden for Y {}

    mod d {
        pub trait Hidden {}
    }
}

pub mod e {
    pub trait Sealed: self::f::Hidden {
        fn foo() {}
    }

    struct X;
    impl self::f::Hidden for X {}

    struct Y;
    impl self::f::Hidden for Y {}
    impl<T: self::f::Hidden> Sealed for T {}

    mod f {
        pub trait Hidden {}
    }
}

struct S;
impl a::Sealed for S {} //~ ERROR the trait bound
impl c::Sealed for S {} //~ ERROR the trait bound
impl e::Sealed for S {} //~ ERROR the trait bound
fn main() {}
