//@ normalize-stderr-test: "(?s)^warning: integer-to-pointer cast.*warning: 1 warning emitted\n\n$" -> ""
use std::mem::{self, MaybeUninit};

#[repr(C)]
struct PointerAtOffset0<'a> {
    ptr: &'a u8,
}

#[repr(C)]
struct PointerAfterBytes<'a> {
    bytes: [u8; 3],
    ptr: &'a u8,
}

#[repr(C)]
struct PointerAtEnd<'a> {
    byte: u8,
    ptr: &'a u8,
}

#[repr(C)]
struct UninitAroundPointer<'a> {
    before: MaybeUninit<[u8; 2]>,
    ptr: &'a u8,
    after: MaybeUninit<[u8; 2]>,
}

#[repr(C)]
struct IntegerAndPointer<'a> {
    integer: u32,
    ptr: &'a u8,
}

fn main() {
    let target = [10u8, 20];
    let pointer_at_offset0 = PointerAtOffset0 { ptr: &target[0] };
    let pointer_after_bytes = PointerAfterBytes { bytes: [1, 2, 3], ptr: &target[0] };
    let pointer_at_end = PointerAtEnd { byte: 4, ptr: &target[0] };
    let uninit_around_pointer = UninitAroundPointer {
        before: MaybeUninit::uninit(),
        ptr: &target[0],
        after: MaybeUninit::uninit(),
    };
    let integer_and_pointer = IntegerAndPointer { integer: 0x44332211, ptr: &target[1] };

    // Keep only fewer than pointer-sized bytes from a pointer representation.
    let fixed_addr_ptr = std::ptr::with_exposed_provenance::<u8>(0x1234);
    let short_pointer_bytes = unsafe {
        let mut bytes = MaybeUninit::<[u8; 1]>::uninit();
        std::ptr::copy_nonoverlapping(
            (&fixed_addr_ptr as *const *const u8).cast::<u8>(),
            bytes.as_mut_ptr().cast::<u8>(),
            mem::size_of::<[u8; 1]>(),
        );
        bytes.assume_init()
    };

    // Break here so all locals are still available to the debugger.
    std::hint::black_box((
        &pointer_at_offset0,
        &pointer_after_bytes,
        &pointer_at_end,
        &uninit_around_pointer,
        &integer_and_pointer,
        &fixed_addr_ptr,
        &short_pointer_bytes,
    ));
}
