//@ compile-flags:-Z unstable-options --extern-html-root-url core=../ --extern-html-root-takes-precedence --generate-link-to-definition

// At depth 1 (top-level), the href should be ../core/...
//@ has extern_html_root_url_relative/index.html
//@ has - '//a/@href' '../core/iter/index.html'
#[doc(no_inline)]
pub use std::iter;

// At depth 2 (inside a module), the href should be ../../core/...
pub mod nested {
    //@ has extern_html_root_url_relative/nested/index.html
    //@ has - '//a/@href' '../../core/future/index.html'
    #[doc(no_inline)]
    pub use std::future;
}

// Also depth 2, but for an intra-doc link.
//@ has extern_html_root_url_relative/intra_doc_link/index.html
//@ has - '//a/@href' '../../core/ptr/fn.write.html'
/// [write](<core::ptr::write()>)
pub mod intra_doc_link {
}

// link-to-definition
//@ has src/extern_html_root_url_relative/extern-html-root-url-relative.rs.html
//@ has - '//a/@href' '../../core/iter/index.html'
//@ has - '//a/@href' '../../core/future/index.html'
