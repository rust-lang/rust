//@ignore-target-windows: No libc on Windows

#![feature(pointer_is_aligned)]
#![feature(strict_provenance)]

use core::ptr;

fn main() {
    // A normal allocation.
    unsafe {
        let mut ptr: *mut libc::c_void = ptr::null_mut();
        let align = 8;
        let size = 64;
        assert_eq!(libc::posix_memalign(&mut ptr, align, size), 0);
        assert!(!ptr.is_null());
        assert!(ptr.is_aligned_to(align));
        ptr.cast::<u8>().write_bytes(1, size);
        libc::free(ptr);
    }

    // Align > size.
    unsafe {
        let mut ptr: *mut libc::c_void = ptr::null_mut();
        let align = 64;
        let size = 8;
        assert_eq!(libc::posix_memalign(&mut ptr, align, size), 0);
        assert!(!ptr.is_null());
        assert!(ptr.is_aligned_to(align));
        ptr.cast::<u8>().write_bytes(1, size);
        libc::free(ptr);
    }

    // Size not multiple of align
    unsafe {
        let mut ptr: *mut libc::c_void = ptr::null_mut();
        let align = 16;
        let size = 31;
        assert_eq!(libc::posix_memalign(&mut ptr, align, size), 0);
        assert!(!ptr.is_null());
        assert!(ptr.is_aligned_to(align));
        ptr.cast::<u8>().write_bytes(1, size);
        libc::free(ptr);
    }

    // Size == 0
    unsafe {
        let mut ptr: *mut libc::c_void = ptr::null_mut();
        let align = 64;
        let size = 0;
        assert_eq!(libc::posix_memalign(&mut ptr, align, size), 0);
        // We are not required to return null if size == 0, but we currently do.
        // It's fine to remove this assert if we start returning non-null pointers.
        assert!(ptr.is_null());
        assert!(ptr.is_aligned_to(align));
        // Regardless of what we return, it must be `free`able.
        libc::free(ptr);
    }

    // Non-power of 2 align
    unsafe {
        let mut ptr: *mut libc::c_void = ptr::without_provenance_mut(0x1234567);
        let align = 15;
        let size = 8;
        assert_eq!(libc::posix_memalign(&mut ptr, align, size), libc::EINVAL);
        // The pointer is not modified on failure, posix_memalign(3) says:
        // > On Linux (and other systems), posix_memalign() does  not  modify  memptr  on failure.
        // > A requirement standardizing this behavior was added in POSIX.1-2008 TC2.
        assert_eq!(ptr.addr(), 0x1234567);
    }

    // Too small align (smaller than ptr)
    unsafe {
        let mut ptr: *mut libc::c_void = ptr::without_provenance_mut(0x1234567);
        let align = std::mem::size_of::<usize>() / 2;
        let size = 8;
        assert_eq!(libc::posix_memalign(&mut ptr, align, size), libc::EINVAL);
        // The pointer is not modified on failure, posix_memalign(3) says:
        // > On Linux (and other systems), posix_memalign() does  not  modify  memptr  on failure.
        // > A requirement standardizing this behavior was added in POSIX.1-2008 TC2.
        assert_eq!(ptr.addr(), 0x1234567);
    }
}
