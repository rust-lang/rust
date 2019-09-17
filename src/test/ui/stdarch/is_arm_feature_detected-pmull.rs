// only-arm

#[cfg(target_arch = "arm")]
fn main() {
    is_arm_feature_detected!("pmull");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "arm"))]
fn main() {}
