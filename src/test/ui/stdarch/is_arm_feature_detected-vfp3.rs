// only-arm

#[cfg(target_arch = "arm")]
fn main() {
    is_arm_feature_detected!("vfp3");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "arm"))]
fn main() {}
