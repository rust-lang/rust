mod a {
    pub fn foo() {}
}

mod b {
    pub fn foo() {}
}

mod c {
    pub use crate::a::foo;
}

mod d {
    use crate::a::foo;
    use crate::a::foo; //~ ERROR the name `foo` is defined multiple times
}

mod e {
    pub use crate::a::*;
    pub use crate::c::*; // ok
}

mod f {
    pub use crate::a::*;
    pub use crate::b::*;
}

mod g {
    pub use crate::a::*;
    pub use crate::f::*;
}

fn main() {
    e::foo();
    f::foo(); //~ ERROR `foo` is ambiguous
    g::foo();
    //~^ ERROR `foo` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
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
