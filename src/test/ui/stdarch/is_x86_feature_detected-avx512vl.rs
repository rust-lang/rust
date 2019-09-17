// only-x86|x86_64

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    is_x86_feature_detected!("avx512vl");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {}
