//@ only-x86_64
//@ compile-flags:--test -C target-feature=+avx
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

#![feature(doc_cfg)]

/// (written on a spider's web) Some Struct
///
/// ```
/// panic!("oh no");
/// ```
#[doc(cfg(target_feature = "avx"))]
pub struct SomeStruct;
