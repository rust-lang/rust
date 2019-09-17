// only-arm

#[cfg(target_arch = "arm")]
fn main() {
    // test for unknown features
    is_arm_feature_detected!("foobar");
    //~^ ERROR use of unstable library feature
}


#[cfg(not(target_arch = "arm"))]
fn main() {}
