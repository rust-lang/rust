// compile-flags: --no-defaults

#![crate_name = "foo"]

// @has foo/a/index.html '//code' 'use *;'
mod a {
    use *;
}

// @has foo/b/index.html '//code' 'pub use *;'
pub mod b {
    pub use *;
}
