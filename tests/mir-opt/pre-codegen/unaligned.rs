//@ compile-flags: -O -Zmir-opt-level=2
//@ ignore-std-debug-assertions (there's one in `ptr::read` via `MaybeUninit::assume_init`)

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
    // CHECK: copy _1 as *const u8 (PtrToPtr);
    // CHECK: &raw mut
    // CHECK: copy_nonoverlapping{{.+}}count = const <T as std::mem::SizedTypeProperties>::SIZE
    // CHECK: &raw const
    // CHECK: copy _2 as *mut u8 (PtrToPtr);
    // CHECK: copy_nonoverlapping{{.+}}count = const <T as std::mem::SizedTypeProperties>::SIZE
    unsafe {
        let val = std::ptr::read_unaligned(src);
        std::ptr::write_unaligned(dst, val);
    }
}
