#![crate_name = "foo"]

// @has foo/index.html '//a[@href="../src/foo/thread-local-src.rs.html#1-6"]' 'source'

// @has foo/constant.FOO.html '//a[@href="../src/foo/thread-local-src.rs.html#6"]' 'source'
thread_local!(pub static FOO: bool = false);
