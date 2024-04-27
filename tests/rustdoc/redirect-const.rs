#![crate_name="foo"]

pub use hidden::STATIC_FOO;
pub use hidden::CONST_FOO;

mod hidden {
    // @has foo/hidden/static.STATIC_FOO.html
    // @has - '//p/a' '../../foo/static.STATIC_FOO.html'
    pub static STATIC_FOO: u64 = 0;
    // @has foo/hidden/constant.CONST_FOO.html
    // @has - '//p/a' '../../foo/constant.CONST_FOO.html'
    pub const CONST_FOO: u64 = 0;
}
