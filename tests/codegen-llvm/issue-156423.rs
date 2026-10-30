//@ compile-flags: -Copt-level=3 --target x86_64-unknown-linux-gnu
//@ only-x86_64-unknown-linux-gnu
//@ min-llvm-version: 23
//@ needs-llvm-components: x86

#![crate_type = "rlib"]
type Result<T> = core::result::Result<T, u32>;
fn map<T, F, U>(value: Result<T>, op: F) -> Result<U>
where
    F: FnOnce(T) -> U,
{
    value.map(op)
}

// CHECK-LABEL: @assign
// CHECK-NOT: @llvm.fshl
// CHECK-NOT: @llvm.fshr
// CHECK: store i32 %0, ptr %out, align 4
// CHECK: ret { i32, i32 } { i32 0, i32 undef }

#[no_mangle]
pub fn assign(val: [u8; 4], out: &mut u32) -> Result<()> {
    let val = map(Ok(val), u32::from_le_bytes)?;

    *out = val;

    Ok(())
}
