// only-aarch64

#[cfg(target_arch = "aarch64")]
fn main() {
    // test for unknown features
    is_aarch64_feature_detected!("foobar");
    //~^ ERROR use of unstable library feature
}


#[cfg(not(target_arch = "aarch64"))]
fn main() {}
