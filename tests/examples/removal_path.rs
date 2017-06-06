pub mod old {
    pub mod a {
        pub struct Abc;
    }

    pub mod b {
        pub use old::a::*;
    }

    pub mod c {
        pub use old::a::Abc;
    }
}

pub mod new {
    pub mod a {
        pub struct Abc;
    }
}
