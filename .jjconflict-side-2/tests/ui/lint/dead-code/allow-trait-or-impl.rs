#![deny(dead_code)]

pub mod a {
    pub trait Foo { }
    impl Foo for u32 { }

    struct PrivateType; //~ ERROR struct `PrivateType` is never constructed
    impl Foo for PrivateType { } // <-- warns as dead, even though Foo is public

    struct AnotherPrivateType; //~ ERROR struct `AnotherPrivateType` is never constructed
    impl Foo for AnotherPrivateType { } // <-- warns as dead, even though Foo is public
}

pub mod b {
    #[allow(dead_code)]
    pub trait Foo { }
    impl Foo for u32 { }

    struct PrivateType;
    impl Foo for PrivateType { } // <-- no warning, trait is "allowed"

    struct AnotherPrivateType;
    impl Foo for AnotherPrivateType { } // <-- no warning, trait is "allowed"
}

pub mod c {
    pub trait Foo { }
    impl Foo for u32 { }

    struct PrivateType;
    #[allow(dead_code)]
    impl Foo for PrivateType { } // <-- no warning, impl is allowed

    struct AnotherPrivateType; //~ ERROR struct `AnotherPrivateType` is never constructed
    impl Foo for AnotherPrivateType { } // <-- warns as dead, even though Foo is public
}

fn main() {}
