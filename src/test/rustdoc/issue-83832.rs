#![crate_name = "foo"]

pub mod io {
    #[deprecated(since = "0.1.8", note = "Use bar() instead")]
    pub trait Reader {}
    pub trait Writer {}
}

// @has foo/mod1/index.html
pub mod mod1 {
    // @has - '//code' 'pub use io::Reader;'
    // @has - '//span' 'Deprecated'
    pub use io::Reader;
}

// @has foo/mod2/index.html
pub mod mod2 {
    // @has - '//code' 'pub use io::Writer;'
    // @!has - '//span' 'Deprecated'
    pub use io::Writer;
}
