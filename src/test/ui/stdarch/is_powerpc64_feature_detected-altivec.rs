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

#[cfg(target_arch = "powerpc64")]
fn main() {
    is_powerpc64_feature_detected!("altivec");
    //~^ ERROR use of unstable library feature
}

#[cfg(not(target_arch = "powerpc64"))]
fn main() {}
