// only-powerpc

#[cfg(target_arch = "powerpc")]
fn main() {
    // test for unknown features
    is_powerpc_feature_detected!("foobar");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "powerpc"))]
fn main() {}
