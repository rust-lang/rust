#![crate_name = "foo"]
#![doc(extern_html_root_url(std = "https://doc.rust-lang.org/nightly"))]

// @has 'foo/index.html'
// @has - '//a[@class="mod"]/@href' 'https://doc.rust-lang.org/nightly/std/index.html'
pub use std as other;
