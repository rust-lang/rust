mod a {
    pub fn foo() {}
}

mod b {
    pub fn foo() {}
}

mod c {
    pub use a::foo;
}

mod d {
    use a::foo;
    use a::foo; //~ ERROR the name `foo` is defined multiple times
}

mod e {
    pub use a::*;
    pub use c::*; // ok
}

mod f {
    pub use a::*;
    pub use b::*;
}

mod g {
    pub use a::*;
    pub use f::*;
}

fn main() {
    e::foo();
    f::foo(); //~ ERROR `foo` is ambiguous
    g::foo();
}

mod ambiguous_module_errors {
    pub mod m1 { pub use super::m1 as foo; pub fn bar() {} }
    pub mod m2 { pub use super::m2 as foo; }

    use self::m1::*;
    use self::m2::*;

    use self::foo::bar; //~ ERROR `foo` is ambiguous

    fn f() {
        foo::bar(); //~ ERROR `foo` is ambiguous
    }
}
