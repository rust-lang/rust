// ignore-s390x
// ignore-emscripten
// ignore-powerpc
// ignore-sparc
// ignore-sparc64
// ignore-mips
// ignore-mips64
// ignore-arm
// ignore-aarch64
// ignore-x86
// ignore-x86_64

#[cfg(any(target_arch = "powerpc64", target_arch = "powerpc64le"))]
fn main() {
    // test for unknown features
    is_powerpc64_feature_detected!("foobar");
    //~^ ERROR use of unstable library feature
}

#[cfg(any(target_arch = "powerpc64", target_arch = "powerpc64le"))]
fn main() {}
