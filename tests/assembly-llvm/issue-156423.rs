//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target x86_64-unknown-linux-gnu
//@ only-x86_64-unknown-linux-gnu
//@ min-llvm-version: 23
//@ needs-llvm-components: x86

#![crate_type = "rlib"]
type Result<T> = core::result::Result<T, u32>;
fn map<T, F, U>(value: Result<T>, op: F) -> Result<U> where F: FnOnce(T) -> U {
    value.map(op)
}

// CHECK-LABEL: assign
// CHECK-NOT: shrl $24, %eax
// CHECK-NOT: shll $8, %edi
// CHECK-NOT: shrdl $8, %eax, %edi
// CHECK: movl %edi, (%rsi)
// CHECK: xorl %eax, %eax
#[no_mangle]
pub fn assign(val: [u8; 4], out: &mut u32) -> Result<()> {
    let val = map(Ok(val), u32::from_le_bytes)?;

    *out = val;

    Ok(())
}
