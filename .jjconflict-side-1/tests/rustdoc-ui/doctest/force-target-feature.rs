//@ only-x86_64
//@ only-linux
//@ compile-flags:--test -C target-feature=+avx
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "rust_out::main::.+" -> "rust_out::main::$$PATH"
//@ failure-status: 101

#![feature(doc_cfg)]

/// (written on a spider's web) Some Struct
///
/// ```
/// panic!("oh no");
/// ```
#[doc(cfg(target_feature = "avx"))]
pub struct SomeStruct;
