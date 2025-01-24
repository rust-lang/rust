use super::alloc::{User, copy_from_userspace, copy_to_userspace};

#[test]
fn test_copy_to_userspace_function() {
    let mut src = [0u8; 100];
    let mut dst = User::<[u8]>::uninitialized(100);

    for i in 0..src.len() {
        src[i] = i as _;
    }

    for size in 0..48 {
        // For all possible alignment
        for offset in 0..8 {
            // overwrite complete dst
            dst.copy_from_enclave(&[0u8; 100]);

            // Copy src[0..size] to dst + offset
            unsafe { copy_to_userspace(src.as_ptr(), dst.as_mut_ptr().add(offset), size) };

            // Verify copy
            for byte in 0..size {
                unsafe {
                    assert_eq!(*dst.as_ptr().add(offset + byte), src[byte as usize]);
                }
            }
        }
    }
}

#[test]
fn test_copy_from_userspace_function() {
    let mut dst = [0u8; 100];
    let mut src = User::<[u8]>::uninitialized(100);

    src.copy_from_enclave(&[0u8; 100]);

    for size in 0..48 {
        // For all possible alignment
        for offset in 0..8 {
            // overwrite complete dst
            dst = [0u8; 100];

            // Copy src[0..size] to dst + offset
            unsafe { copy_from_userspace(src.as_ptr().offset(offset), dst.as_mut_ptr(), size) };

            // Verify copy
            for byte in 0..size {
                unsafe {
                    assert_eq!(dst[byte as usize], *src.as_ptr().offset(offset + byte as isize));
                }
            }
        }
    }
}
