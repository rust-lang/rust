// Checks that "jump-to-def" links on source pages don't generate links with `\` as separator
// on Windows. For other systems, just checks that the links are valid.

//@ compile-flags: -Zunstable-options --generate-link-to-definition

#![crate_name = "foo"]

//@ has 'src/foo/nested-dirs/a/b/mod.rs.html'
//@ has - '//pre[@class="rust"]//a[@href="../../../nested-dirs.rs.html#14"]' 'Top'
//@ has - '//pre[@class="rust"]//a[@href="../mod.rs.html#3"]' 'Middle'
#[path = "nested-dirs/a/mod.rs"]
pub mod a;

pub struct Top;
