// compile-flags: -Cmetadata=aux

pub mod tree {
    pub use tree;
}

pub mod tree2 {
    pub mod prelude {
        pub use tree2;
    }
}
