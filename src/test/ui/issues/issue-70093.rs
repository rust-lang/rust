// run-pass
// compile-flags: -Zlink-native-libraries=no -Cdefault-linker-libraries=yes

#[link(name = "some-random-non-existent-library", kind = "static")]
extern "C" {}

fn main() {}
