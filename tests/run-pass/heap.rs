#![feature(box_syntax)]
#![feature(allocator_api)]

use std::alloc::{Global, Alloc, Layout, System};

fn make_box() -> Box<(i16, i16)> {
    Box::new((1, 2))
}

fn make_box_syntax() -> Box<(i16, i16)> {
    box (1, 2)
}

fn allocate_reallocate() {
    let mut s = String::new();

    // 6 byte heap alloc (__rust_allocate)
    s.push_str("foobar");
    assert_eq!(s.len(), 6);
    assert_eq!(s.capacity(), 6);

    // heap size doubled to 12 (__rust_reallocate)
    s.push_str("baz");
    assert_eq!(s.len(), 9);
    assert_eq!(s.capacity(), 12);

    // heap size reduced to 9  (__rust_reallocate)
    s.shrink_to_fit();
    assert_eq!(s.len(), 9);
    assert_eq!(s.capacity(), 9);
}

fn check_overalign_requests<T: Alloc>(mut allocator: T) {
    let size = 8;
    let align = 16; // greater than size
    let iterations = 1; // Miri is deterministic, no need to try many times
    unsafe {
        let pointers: Vec<_> = (0..iterations).map(|_| {
            allocator.alloc(Layout::from_size_align(size, align).unwrap()).unwrap()
        }).collect();
        for &ptr in &pointers {
            assert_eq!((ptr.as_ptr() as usize) % align, 0,
                       "Got a pointer less aligned than requested")
        }

        // Clean up
        for &ptr in &pointers {
            allocator.dealloc(ptr, Layout::from_size_align(size, align).unwrap())
        }
    }
}

fn main() {
    assert_eq!(*make_box(), (1, 2));
    assert_eq!(*make_box_syntax(), (1, 2));
    allocate_reallocate();

    check_overalign_requests(System);
    check_overalign_requests(Global);
}
