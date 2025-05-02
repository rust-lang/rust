//@ assembly-output: emit-asm
//@ only-x86_64
//@ ignore-sgx Test incompatible with LVI mitigations
//@ compile-flags: -C opt-level=3
//! Ensure that indexing a slice with `bool` does not
//! generate any redundant `jmp` and `and` instructions.
//! Discovered in issue #123216.

#![crate_type = "lib"]

#[no_mangle]
fn f(a: u32, b: bool, c: bool, d: &mut [u128; 2]) {
    // CHECK-LABEL: f:
    // CHECK: testl  %esi, %esi
    // CHECK: je
    // CHECK: xorb   %dl, %dil
    // CHECK: orb    $1, (%rcx)
    // CHECK: movzbl %dil, %eax
    // CHECK: andl   $1, %eax
    // CHECK: shll   $4, %eax
    // CHECK: orb    $1, (%rcx,%rax)
    // CHECK-NOT:     jmp
    // CHECK-NOT:     andl %dil, $1
    // CHECK: retq
    let mut a = a & 1 != 0;
    if b {
        a ^= c;
        d[0] |= 1;
    }
    d[a as usize] |= 1;
}
