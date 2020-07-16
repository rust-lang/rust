// check-pass
// revisions: public private
// [private]compile-flags: --document-private-items
#![cfg_attr(private, deny(intra_doc_link_resolution_failure))]

/// docs [DontDocMe]
//[public]~^ WARNING `[DontDocMe]` public documentation for `DocMe` links to a private item
// FIXME: for [private] we should also make sure the link was actually generated
pub struct DocMe;
struct DontDocMe;
