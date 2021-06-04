// run-pass
// compile-flags: -Zlink-native-libraries=no -Cdefault-linker-libraries=yes
// ignore-windows - this will probably only work on unixish systems
// ignore-fuchsia - missing __libc_start_main for some reason (#84733)

#[link(name = "some-random-non-existent-library", kind = "static")]
extern "C" {}

fn main() {}
