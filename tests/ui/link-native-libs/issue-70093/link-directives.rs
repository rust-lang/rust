// Ensure that `#[link]` attributes are entirely ignore when using `-Zlink-directives=no`.

//@ run-pass
//@ compile-flags: -Zlink-directives=no
//@ ignore-fuchsia - missing __libc_start_main for some reason (#84733)
//@ ignore-cross-compile - default-linker-libraries=yes doesn't play well with cross compiling

// Usually these `#[link]` attribute would cause `libsome-random-non-existent-library`
// to be passed to the linker, causing it to fail because the file doesn't exist.
// However, with -Zlink-directives=no, the `#[link]` is ignored.
#[link(name = "some-random-non-existent-library", kind = "static")]
extern "C" {}

fn main() {}
