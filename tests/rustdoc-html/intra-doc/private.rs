//@ compile-flags: --document-private-items

// make sure to update `rustdoc-ui/intra-doc/private.rs` if you update this file

#![allow(rustdoc::private_intra_doc_links)]

#![crate_name = "private"]

/// docs [DontDocMe] [DontDocMe::f] [DontDocMe::x]
//@ has private/struct.DocMe.html '//*a[@href="struct.DontDocMe.html"]' 'DontDocMe'
//@ has private/struct.DocMe.html '//*a[@href="struct.DontDocMe.html#method.f"]' 'DontDocMe::f'
//@ has private/struct.DocMe.html '//*a[@href="struct.DontDocMe.html#structfield.x"]' 'DontDocMe::x'
pub struct DocMe;
struct DontDocMe {
    x: usize,
}

impl DontDocMe {
    fn f() {}
}
