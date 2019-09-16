// ignore-s390x
// ignore-emscripten
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-sparc
// ignore-sparc64
// ignore-mips
// ignore-mips64
// ignore-arm
// ignore-aarch64

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn main() {
    // test for unknown features
    is_x86_feature_detected!("foobar");
    //~^ ERROR unknown x86 target feature: foobar
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {}
