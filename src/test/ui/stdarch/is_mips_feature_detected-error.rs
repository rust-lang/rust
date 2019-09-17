// only-mips

#[cfg(target_arch = "mips")]
fn main() {
    // test for unknown features
    is_mips_feature_detected!("foobar");
    //~^ ERROR use of unstable library feature
}


#[cfg(not(target_arch = "mips"))]
fn main() {}
