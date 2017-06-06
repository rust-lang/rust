pub mod old {
    pub mod a {
        pub struct Abc;
    }

    pub use old::a as b;
}

pub mod new {
    pub mod a {
        pub struct Abc;
    }

    pub mod b {
        pub use new::a::*;
    }
}
