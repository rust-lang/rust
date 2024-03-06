// Tests #108655: closures in `#[target_feature]` functions can still be marked #[inline(always)]

//@ check-pass
//@ only-x86_64

#![feature(target_feature_11)]

#[target_feature(enable = "avx")]
pub unsafe fn test() {
    ({
        #[inline(always)]
        move || {}
    })();
}

fn main() {}
