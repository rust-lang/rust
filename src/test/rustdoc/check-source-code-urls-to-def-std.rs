// compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

// @has 'src/foo/check-source-code-urls-to-def-std.rs.html'

fn babar() {}

// @has - '//a[@href="{{channel}}/std/primitive.u32.html"]' 'u32'
// @has - '//a[@href="{{channel}}/std/primitive.str.html"]' 'str'
// @has - '//a[@href="{{channel}}/std/primitive.bool.html"]' 'bool'
// @has - '//a[@href="../../src/foo/check-source-code-urls-to-def-std.rs.html#7"]' 'babar'
pub fn foo(a: u32, b: &str, c: String) {
    let x = 12;
    let y: bool = true;
    babar();
}
