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
    is_x86_feature_detected!("f16c");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn main() {}
