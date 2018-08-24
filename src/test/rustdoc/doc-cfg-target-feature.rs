// only-x86_64
// compile-flags:--test
// should-fail
// no-system-llvm

// #49723: rustdoc didn't add target features when extracting or running doctests

#![feature(doc_cfg)]

/// Foo
///
/// # Examples
///
/// ```
/// #![feature(cfg_target_feature)]
///
/// #[cfg(target_feature = "sse")]
/// assert!(false);
/// ```
#[doc(cfg(target_feature = "sse"))]
pub unsafe fn foo() {}
