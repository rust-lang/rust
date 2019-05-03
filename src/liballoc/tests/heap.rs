use std::alloc::{GlobalAlloc, Layout, System};

/// Issue #45955.
#[test]
fn alloc_system_overaligned_request() {
    check_overalign_requests(System)
}

fn check_overalign_requests<T: GlobalAlloc>(allocator: T) {
    let size = 8;
    let align = 16; // greater than size
    let iterations = 100;
    unsafe {
        let pointers: Vec<_> = (0..iterations).map(|_| {
            allocator.alloc(Layout::from_size_align(size, align).unwrap())
        }).collect();
        for &ptr in &pointers {
            assert_eq!((ptr as usize) % align, 0,
                       "Got a pointer less aligned than requested")
        }

        // Clean up
        for &ptr in &pointers {
            allocator.dealloc(ptr, Layout::from_size_align(size, align).unwrap())
        }
    }
}
