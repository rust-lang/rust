// check-pass

pub mod upstream_a {
    mod inner {
        pub struct Foo {}
        pub struct Bar {}
    }

    struct Foo;
    //~^ WARN private item shadows public glob re-export

    pub use self::inner::*;
}

pub mod upstream_b {
    mod inner {
        pub struct Foo {}
        pub struct Qux {}
    }

    mod other {
        pub struct Foo;
    }

    pub use self::inner::*;

    use self::other::Foo;
    //~^ WARN private item shadows public glob re-export
}

pub mod upstream_c {
    mod no_def_id {
        #![allow(non_camel_case_types)]
        pub struct u8;
        pub struct World;
    }

    pub use self::no_def_id::*;

    use std::primitive::u8;
    //~^ WARN private item shadows public glob re-export
}

// Downstream crate
// mod downstream {
//     fn proof() {
//         let _ = crate::upstream_a::Foo;
//         let _ = crate::upstream_b::Foo;
//     }
// }

pub fn main() {}
