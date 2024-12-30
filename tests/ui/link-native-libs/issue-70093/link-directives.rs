//@ run-pass
//@ compile-flags: -Zlink-directives=no
//@ ignore-fuchsia - missing __libc_start_main for some reason (#84733)
//@ ignore-cross-compile - default-linker-libraries=yes doesn't play well with cross compiling

#[link(name = "some-random-non-existent-library", kind = "static")]
extern "C" {}

fn main() {}
