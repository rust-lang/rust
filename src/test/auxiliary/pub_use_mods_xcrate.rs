pub mod a {
    pub mod b {
        pub mod c {
            fn f(){}
            fn g(){}
        }
    }

    pub use b::c;
}

