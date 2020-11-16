// aux-crate: priv:extern_html_root_url=extern-html-root-url.rs
// compile-flags: -Z unstable-options
// compile-flags: --edition 2018
// compile-flags: --extern-html-root-url libextern_html_root_url.so=https://example.com/core/0.1.0

// @has extern_html_root_url/index.html
// @has - '//a/@href' 'https://example.com/core/0.1.0/extern_html_root_url/iter/index.html'
#[doc(no_inline)]
pub use extern_html_root_url::iter;
