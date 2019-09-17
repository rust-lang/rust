// only-mips64

#[cfg(target_arch = "mips64")]
fn main() {
    is_mips_feature_detected!("msa");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "mips64"))]
fn main() {}
