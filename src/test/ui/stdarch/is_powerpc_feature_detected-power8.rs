// ignore-s390x
// ignore-emscripten
// ignore-powerpc64
// ignore-powerpc64le
// ignore-sparc
// ignore-sparc64
// ignore-mips
// ignore-mips64
// ignore-arm
// ignore-aarch64
// ignore-x86
// ignore-x86_64

#[cfg(target_arch = "powerpc")]
fn main() {
    is_powerpc_feature_detected!("power8");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "powerpc"))]
fn main() {}
