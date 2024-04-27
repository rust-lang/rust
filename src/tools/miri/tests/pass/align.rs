//@compile-flags: -Zmiri-permissive-provenance

/// This manually makes sure that we have a pointer with the proper alignment.
fn manual_alignment() {
    let x = &mut [0u8; 3];
    let base_addr = x as *mut _ as usize;
    let base_addr_aligned = if base_addr % 2 == 0 { base_addr } else { base_addr + 1 };
    let u16_ptr = base_addr_aligned as *mut u16;
    unsafe {
        *u16_ptr = 2;
    }
}

/// Test standard library `align_to`.
fn align_to() {
    const LEN: usize = 128;
    let buf = &[0u8; LEN];
    let (l, m, r) = unsafe { buf.align_to::<i32>() };
    assert!(m.len() * 4 >= LEN - 4);
    assert!(l.len() + r.len() <= 4);
}

fn main() {
    // Do this a couple times in a loop because it may work "by chance".
    for _ in 0..20 {
        manual_alignment();
        align_to();
    }
}
