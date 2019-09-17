// only-powerpc64|powerpc64le

#[cfg(any(target_arch = "powerpc64", target_arch = "powerpc64le"))]
fn main() {
    // test for unknown features
    is_powerpc64_feature_detected!("foobar");
    //~^ ERROR use of unstable library feature
}

#[cfg(any(target_arch = "powerpc64", target_arch = "powerpc64le"))]
fn main() {}
