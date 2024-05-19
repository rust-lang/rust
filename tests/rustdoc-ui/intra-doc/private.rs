//@ check-pass
//@ revisions: public private
//@ [private]compile-flags: --document-private-items

// make sure to update `rustdoc/intra-doc/private.rs` if you update this file

/// docs [DontDocMe] [DontDocMe::f] [DontDocMe::x]
//~^ WARNING public documentation for `DocMe` links to private item `DontDocMe`
//~| WARNING public documentation for `DocMe` links to private item `DontDocMe::x`
//~| WARNING public documentation for `DocMe` links to private item `DontDocMe::f`
pub struct DocMe;
struct DontDocMe {
    x: usize,
}

impl DontDocMe {
    fn f() {}
}
