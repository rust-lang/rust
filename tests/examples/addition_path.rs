pub mod old {
    pub mod a {
        pub struct Abc;
    }
}

pub mod new {
    pub mod a {
        pub struct Abc;
    }

    pub mod b {
        pub use new::a::*;
    }

    pub mod c {
        pub use new::a::Abc;
    }
}
