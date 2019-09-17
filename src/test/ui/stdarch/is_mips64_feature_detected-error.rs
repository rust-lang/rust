// only-mips64

#[cfg(target_arch = "mips64")]
fn main() {
    // test for unknown features
    is_mips64_feature_detected!("foobar");
    //~^ ERROR use of unstable library feature
}


#[cfg(not(target_arch = "mips64"))]
fn main() {}
