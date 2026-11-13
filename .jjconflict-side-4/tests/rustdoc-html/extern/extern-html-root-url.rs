//@ compile-flags:-Z unstable-options --extern-html-root-url html_root=https://example.com/override --extern-html-root-url no_html_root=https://example.com/override
//@ aux-build:html_root.rs
//@ aux-build:no_html_root.rs
// NOTE: intentionally does not build any auxiliary docs

extern crate html_root;
extern crate no_html_root;

//@ has extern_html_root_url/index.html
// `html_root_url` should override `--extern-html-root-url`
//@ has - '//a/@href' 'https://example.com/html_root/html_root/fn.foo.html'
#[doc(no_inline)]
pub use html_root::foo;

#[doc(no_inline)]
// `--extern-html-root-url` should apply if no `html_root_url` is given
//@ has - '//a/@href' 'https://example.com/override/no_html_root/fn.bar.html'
pub use no_html_root::bar;
