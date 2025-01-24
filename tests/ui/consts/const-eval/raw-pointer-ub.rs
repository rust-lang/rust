const MISALIGNED_LOAD: () = unsafe {
    let mem = [0u32; 8];
    let ptr = mem.as_ptr().byte_add(1);
    let _val = *ptr; //~ERROR: evaluation of constant value failed
    //~^NOTE: based on pointer with alignment 1, but alignment 4 is required
};

const MISALIGNED_STORE: () = unsafe {
    let mut mem = [0u32; 8];
    let ptr = mem.as_mut_ptr().byte_add(1);
    *ptr = 0; //~ERROR: evaluation of constant value failed
    //~^NOTE: based on pointer with alignment 1, but alignment 4 is required
};

const MISALIGNED_COPY: () = unsafe {
    let x = &[0_u8; 4];
    let y = x.as_ptr().cast::<u32>();
    let mut z = 123;
    y.copy_to_nonoverlapping(&mut z, 1);
    //~^NOTE
    // The actual error points into the implementation of `copy_to_nonoverlapping`.
};

const MISALIGNED_FIELD: () = unsafe {
    #[repr(align(16))]
    struct Aligned(f32);

    let mem = [0f32; 8];
    let ptr = mem.as_ptr().cast::<Aligned>();
    // Accessing an f32 field but we still require the alignment of the pointer type.
    let _val = (*ptr).0; //~ERROR: evaluation of constant value failed
    //~^NOTE: based on pointer with alignment 4, but alignment 16 is required
};

const OOB: () = unsafe {
    let mem = [0u32; 1];
    let ptr = mem.as_ptr().cast::<u64>();
    let _val = *ptr; //~ERROR: evaluation of constant value failed
    //~^NOTE: expected a pointer to 8 bytes of memory
};

fn main() {}
