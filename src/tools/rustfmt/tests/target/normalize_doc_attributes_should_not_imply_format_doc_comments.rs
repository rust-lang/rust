// rustfmt-normalize_doc_attributes: true

/// Foo
///
/// # Example
/// ```
/// # #![cfg_attr(not(dox), feature(cfg_target_feature, target_feature, stdsimd))]
/// # #![cfg_attr(not(dox), no_std)]
/// fn foo() {  }
/// ```
///
fn foo() {}

///Bar documents
fn bar() {}
