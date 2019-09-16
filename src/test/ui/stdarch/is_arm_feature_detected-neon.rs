// ignore-s390x
// ignore-emscripten
// ignore-powerpc
// ignore-powerpc64
// ignore-powerpc64le
// ignore-sparc
// ignore-sparc64
// ignore-mips
// ignore-mips64
// ignore-aarch64
// ignore-x86
// ignore-x86_64

#[cfg(target_arch = "arm")]
fn main() {
    is_arm_feature_detected!("neon");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "arm"))]
fn main() {}
