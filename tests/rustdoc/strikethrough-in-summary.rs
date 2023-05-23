#![crate_name = "foo"]

// @has foo/index.html '//del' 'strike'

/// ~~strike~~
pub fn strike() {}
