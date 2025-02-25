use std::alloc::{Allocator, Global, Layout, System};

/// Issue #45955 and #62251.
#[test]
fn alloc_system_overaligned_request() {
    check_overalign_requests(System)
}

#[test]
fn std_heap_overaligned_request() {
    check_overalign_requests(Global)
}

fn check_overalign_requests<T: Allocator>(allocator: T) {
    for &align in &[4, 8, 16, 32] {
        // less than and bigger than `MIN_ALIGN`
        for &size in &[align / 2, align - 1] {
            // size less than alignment
            let iterations = 128;
            unsafe {
                let pointers: Vec<_> = (0..iterations)
                    .map(|_| {
                        allocator.allocate(Layout::from_size_align(size, align).unwrap()).unwrap()
                    })
                    .collect();
                for &ptr in &pointers {
                    assert_eq!(
                        (ptr.as_non_null_ptr().as_ptr() as usize) % align,
                        0,
                        "Got a pointer less aligned than requested"
                    )
                }

                // Clean up
                for &ptr in &pointers {
                    allocator.deallocate(
                        ptr.as_non_null_ptr(),
                        Layout::from_size_align(size, align).unwrap(),
                    )
                }
            }
        }
    }
}
