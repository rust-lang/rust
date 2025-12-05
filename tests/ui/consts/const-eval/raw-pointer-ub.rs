const MISALIGNED_LOAD: () = unsafe {
    let mem = [0u32; 8];
    let ptr = mem.as_ptr().byte_add(1);
    let _val = *ptr; //~NOTE: failed here
    //~^ERROR: based on pointer with alignment 1, but alignment 4 is required
};

const MISALIGNED_STORE: () = unsafe {
    let mut mem = [0u32; 8];
    let ptr = mem.as_mut_ptr().byte_add(1);
    *ptr = 0; //~NOTE: failed here
    //~^ERROR: based on pointer with alignment 1, but alignment 4 is required
};

const MISALIGNED_COPY: () = unsafe {
    let x = &[0_u8; 4];
    let y = x.as_ptr().cast::<u32>();
    let mut z = 123;
    y.copy_to_nonoverlapping(&mut z, 1);
    //~^ NOTE failed inside this call
    //~| NOTE inside `std::ptr::copy_nonoverlapping::<u32>`
    //~| ERROR accessing memory with alignment 1, but alignment 4 is required
    // The actual error points into the implementation of `copy_to_nonoverlapping`.
};

const MISALIGNED_FIELD: () = unsafe {
    #[repr(align(16))]
    struct Aligned(f32);

    let mem = [0f32; 8];
    let ptr = mem.as_ptr().cast::<Aligned>();
    // Accessing an f32 field but we still require the alignment of the pointer type.
    let _val = (*ptr).0; //~NOTE: failed here
    //~^ERROR: based on pointer with alignment 4, but alignment 16 is required
};

const OOB: () = unsafe {
    let mem = [0u32; 1];
    let ptr = mem.as_ptr().cast::<u64>();
    let _val = *ptr; //~NOTE: failed here
    //~^ERROR: is only 4 bytes from the end of the allocation
};

fn main() {}
