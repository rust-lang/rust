//@ only-x86_64
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "rust_out::main::.+" -> "rust_out::main::$$PATH"
//@ compile-flags:--test
//@ failure-status: 101

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
