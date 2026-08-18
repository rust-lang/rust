//@ known-bug: #138262
//@ compile-flags: -Zsanitizer=cfi -Ccodegen-units=1 -Clto -Clink-dead-code=true -Cunsafe-allow-abi-mismatch=sanitizer
//@ ignore-backends: gcc
//@ needs-sanitizer-cfi
fn foo<const N: usize>() {}

core::arch::global_asm!("/* {} */", sym foo::<{
    || {};
    0
}>);

fn main() {}
