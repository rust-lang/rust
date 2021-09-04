// compile-flags:-Z unstable-options --extern-html-root-url core=https://example.com/core/0.1.0

// @has extern_html_root_url/index.html
// @has - '//a/@href' 'https://example.com/core/0.1.0/core/iter/index.html'
#[doc(no_inline)]
pub use std::iter;
