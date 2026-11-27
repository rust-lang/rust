//@ assembly-output: emit-asm
//@ only-x86_64
//@ ignore-windows CHECK patterns use the SysV x86-64 calling convention
//@ ignore-sgx Test incompatible with LVI mitigations
//@ compile-flags: -Copt-level=3

//! Regression test for https://github.com/rust-lang/rust/issues/123216.
//! Indexing with a `bool` should not generate redundant `jmp` or `and`
//! instructions.

#![crate_type = "lib"]

#[no_mangle]
pub fn bool_index(a: u32, b: bool, c: bool, d: &mut [u128; 2]) {
    // CHECK-LABEL: bool_index:
    // CHECK: testl  %esi, %esi
    // CHECK: je
    // CHECK: xorb   {{%dl, %dil|%dil, %dl}}
    // CHECK: orb    $1, (%rcx)
    // CHECK-NOT: jmp
    // CHECK-NOT: andb $1, %dil
    // CHECK: movzbl %dil, %eax
    // CHECK: andl   $1, %eax
    // CHECK: shll   $4, %eax
    // CHECK: orb    $1, (%rcx,%rax)
    // CHECK: retq

    let mut a = a & 1 != 0;

    if b {
        a ^= c;
        d[0] |= 1;
    }

    d[a as usize] |= 1;
}
