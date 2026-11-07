//! Ensure that the target features given in the target spec are actually enabled.
//@only-target: armv7

fn main() {
    assert!(cfg!(target_feature = "v7"));
    assert!(cfg!(target_feature = "vfp2"));
    assert!(cfg!(target_feature = "thumb2"));
}
