// ignore-s390x
// ignore-emscripten
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-sparc
// ignore-sparc64
// ignore-mips64
// ignore-arm
// ignore-aarch64
// ignore-x86
// ignore-x86_64

#[cfg(target_arch = "mips")]
fn main() {
    // test for unknown features
    is_mips_feature_detected!("foobar");
    //~^ ERROR use of unstable library feature
}


#[cfg(not(target_arch = "mips"))]
fn main() {}
