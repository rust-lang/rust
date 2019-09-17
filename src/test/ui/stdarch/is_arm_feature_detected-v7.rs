// only-arm

#[cfg(target_arch = "arm")]
fn main() {
    is_arm_feature_detected!("v7");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "arm"))]
fn main() {}
