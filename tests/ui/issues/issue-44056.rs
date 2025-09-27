//@ build-pass (FIXME(55996): should be run on targets supporting avx)
//@ only-x86_64
//@ no-prefer-dynamic
//@ compile-flags: -Ctarget-feature=+avx -Clto
//@ ignore-backends: gcc

fn main() {}
