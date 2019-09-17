// only-powerpc64|powerpc64le

#[cfg(target_arch = "powerpc64")]
fn main() {
    is_powerpc64_feature_detected!("vsx");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "powerpc64"))]
fn main() {}
