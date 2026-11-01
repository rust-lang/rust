//@only-target: x86_64 # uses x86 target features
//@ignore-target: x86_64-apple-darwin # that target actually has ssse3

fn main() {
    assert!(!is_x86_feature_detected!("ssse3"));
    unsafe {
        ssse3_fn(); //~ ERROR: calling a function that requires unavailable target features: ssse3
    }
}

#[target_feature(enable = "ssse3")]
unsafe fn ssse3_fn() {}
