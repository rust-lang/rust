// only-x86|x86_64

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    // test for unknown features
    is_x86_feature_detected!("foobar");
    //~^ ERROR unknown x86 target feature: foobar
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {}
