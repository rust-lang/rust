// only-x86_64
// compile-flags:--test -C target-feature=+avx
// should-fail

/// (written on a spider's web) Some Struct
///
/// ```
/// panic!("oh no");
/// ```
#[doc(cfg(target_feature = "avx"))]
pub struct SomeStruct;
