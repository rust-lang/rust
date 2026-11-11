//@ compile-flags: -O -Zmir-opt-level=2
//@ ignore-std-debug-assertions (there's one in `ptr::read`)

#![crate_type = "lib"]

// EMIT_MIR unaligned.unaligned_copy_manual.runtime-optimized.after.mir
pub unsafe fn unaligned_copy_manual(src: *const u128, dst: *mut u128) {
    #[repr(packed)]
    struct Packed<T>(T);

    // CHECK-LABEL: fn unaligned_copy_manual(_1: *const u128, _2: *mut u128) -> ()
    // CHECK: [[SRCU:_.+]] = copy _1 as *const unaligned_copy_manual::Packed<u128> (PtrToPtr);
    // CHECK: [[DSTU:_.+]] = copy _2 as *mut unaligned_copy_manual::Packed<u128> (PtrToPtr);
    // CHECK: [[TEMP:_.+]] = copy ((*[[SRCU]]).0: u128);
    // ((*[[DSTU]]).0: u128) = move [[TEMP]];
    let src = src.cast::<Packed<u128>>();
    let dst = dst.cast::<Packed<u128>>();
    unsafe { (*dst).0 = (*src).0 };
}

// EMIT_MIR unaligned.unaligned_copy_generic.runtime-optimized.after.mir
pub unsafe fn unaligned_copy_generic<T>(src: *const T, dst: *mut T) {
    // CHECK-LABEL: fn unaligned_copy_generic(
    // CHECK: debug src => _1;
    // CHECK: debug dst => _2;
    // CHECK: debug val => [[VAL:_.+]];
    // CHECK: [[SRC_P:_.+]] = copy _1 as *const {{.+}}::Unaligned<T> (PtrToPtr);
    // CHECK: [[PACKED1:_.+]] = copy (*[[SRC_P]]);
    // CHECK: [[VAL]] = copy [[PACKED1]] as T (Transmute);
    // CHECK: [[DST_P:_.+]] = copy _2 as *mut {{.+}}::Unaligned<T> (PtrToPtr);
    // CHECK: [[PACKED2:_.+]] = {{.+}}::Unaligned::<T>(copy [[VAL]]);
    // CHECK: (*[[DST_P]]) = copy [[PACKED2]];
    // CHECK-NOT: copy_nonoverlapping
    // CHECK-NOT: drop
    unsafe {
        let val = std::ptr::read_unaligned(src);
        std::ptr::write_unaligned(dst, val);
    }
}
