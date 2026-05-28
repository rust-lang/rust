//@only-target: x86_64 # uses x86 target features
//@compile-flags: -C target-feature=+ssse3

fn main() {
    assert!(is_x86_feature_detected!("ssse3"));
    unsafe {
        ssse3_fn();
    }
}

#[target_feature(enable = "ssse3")]
unsafe fn ssse3_fn() {}
