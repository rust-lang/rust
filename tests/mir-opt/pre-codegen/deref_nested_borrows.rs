// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

fn src(x: &&u8) -> bool {
    // CHECK-LABEL: fn src(
    // CHECK-NOT: _0 = const true;
    // CHECK: _0 = Eq({{.*}}, {{.*}});
    // CHECK-NOT: _0 = const true;
    let y = **x;
    unsafe { unknown() };
    **x == y
}

#[inline(never)]
unsafe fn unknown() {
    // CHECK-LABEL: fn unknown(
}

fn main() {
    // CHECK-LABEL: fn main(
    src(&&0);
}

// EMIT_MIR deref_nested_borrows.src.GVN.diff
// EMIT_MIR deref_nested_borrows.src.PreCodegen.after.mir
