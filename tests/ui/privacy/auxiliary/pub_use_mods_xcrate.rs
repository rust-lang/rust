pub mod a {
    pub use a::b::c;

    pub mod b {
        pub mod c {
            fn f(){}
            fn g(){}
        }
    }
}
