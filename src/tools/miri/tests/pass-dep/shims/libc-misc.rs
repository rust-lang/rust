//@ignore-target-windows: No libc on Windows
//@compile-flags: -Zmiri-disable-isolation
#![feature(io_error_more)]
#![feature(pointer_is_aligned_to)]
#![feature(strict_provenance)]

use std::mem::{self, transmute};
use std::ptr;

/// Tests whether each thread has its own `__errno_location`.
fn test_thread_local_errno() {
    #[cfg(any(target_os = "illumos", target_os = "solaris"))]
    use libc::___errno as __errno_location;
    #[cfg(target_os = "linux")]
    use libc::__errno_location;
    #[cfg(any(target_os = "macos", target_os = "freebsd"))]
    use libc::__error as __errno_location;

    unsafe {
        *__errno_location() = 0xBEEF;
        std::thread::spawn(|| {
            assert_eq!(*__errno_location(), 0);
            *__errno_location() = 0xBAD1DEA;
            assert_eq!(*__errno_location(), 0xBAD1DEA);
        })
        .join()
        .unwrap();
        assert_eq!(*__errno_location(), 0xBEEF);
    }
}

fn test_memcpy() {
    unsafe {
        let src = [1i8, 2, 3];
        let dest = libc::calloc(3, 1);
        libc::memcpy(dest, src.as_ptr() as *const libc::c_void, 3);
        let slc = std::slice::from_raw_parts(dest as *const i8, 3);
        assert_eq!(*slc, [1i8, 2, 3]);
        libc::free(dest);
    }

    unsafe {
        let src = [1i8, 2, 3];
        let dest = libc::calloc(4, 1);
        libc::memcpy(dest, src.as_ptr() as *const libc::c_void, 3);
        let slc = std::slice::from_raw_parts(dest as *const i8, 4);
        assert_eq!(*slc, [1i8, 2, 3, 0]);
        libc::free(dest);
    }

    unsafe {
        let src = 123_i32;
        let mut dest = 0_i32;
        libc::memcpy(
            &mut dest as *mut i32 as *mut libc::c_void,
            &src as *const i32 as *const libc::c_void,
            mem::size_of::<i32>(),
        );
        assert_eq!(dest, src);
    }

    unsafe {
        let src = Some(123);
        let mut dest: Option<i32> = None;
        libc::memcpy(
            &mut dest as *mut Option<i32> as *mut libc::c_void,
            &src as *const Option<i32> as *const libc::c_void,
            mem::size_of::<Option<i32>>(),
        );
        assert_eq!(dest, src);
    }

    unsafe {
        let src = &123;
        let mut dest = &42;
        libc::memcpy(
            &mut dest as *mut &'static i32 as *mut libc::c_void,
            &src as *const &'static i32 as *const libc::c_void,
            mem::size_of::<&'static i32>(),
        );
        assert_eq!(*dest, 123);
    }
}

fn test_strcpy() {
    use std::ffi::{CStr, CString};

    // case: src_size equals dest_size
    unsafe {
        let src = CString::new("rust").unwrap();
        let size = src.as_bytes_with_nul().len();
        let dest = libc::malloc(size);
        libc::strcpy(dest as *mut libc::c_char, src.as_ptr());
        assert_eq!(CStr::from_ptr(dest as *const libc::c_char), src.as_ref());
        libc::free(dest);
    }

    // case: src_size is less than dest_size
    unsafe {
        let src = CString::new("rust").unwrap();
        let size = src.as_bytes_with_nul().len();
        let dest = libc::malloc(size + 1);
        libc::strcpy(dest as *mut libc::c_char, src.as_ptr());
        assert_eq!(CStr::from_ptr(dest as *const libc::c_char), src.as_ref());
        libc::free(dest);
    }
}

#[cfg(target_os = "linux")]
fn test_sigrt() {
    let min = libc::SIGRTMIN();
    let max = libc::SIGRTMAX();

    // "The Linux kernel supports a range of 33 different real-time
    // signals, numbered 32 to 64"
    assert!(min >= 32);
    assert!(max >= 32);
    assert!(min <= 64);
    assert!(max <= 64);

    // "POSIX.1-2001 requires that an implementation support at least
    // _POSIX_RTSIG_MAX (8) real-time signals."
    assert!(min < max);
    assert!(max - min >= 8)
}

fn test_dlsym() {
    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, b"notasymbol\0".as_ptr().cast()) };
    assert!(addr as usize == 0);

    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, b"isatty\0".as_ptr().cast()) };
    assert!(addr as usize != 0);
    let isatty: extern "C" fn(i32) -> i32 = unsafe { transmute(addr) };
    assert_eq!(isatty(999), 0);
    let errno = std::io::Error::last_os_error().raw_os_error().unwrap();
    assert_eq!(errno, libc::EBADF);
}

#[cfg(not(any(target_os = "macos", target_os = "illumos")))]
fn test_reallocarray() {
    unsafe {
        let mut p = libc::reallocarray(std::ptr::null_mut(), 4096, 2);
        assert!(!p.is_null());
        libc::free(p);
        p = libc::malloc(16);
        let r = libc::reallocarray(p, 2, 32);
        assert!(!r.is_null());
        libc::free(r);
    }
}

fn test_memalign() {
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

fn main() {
    test_thread_local_errno();

    test_dlsym();

    test_memcpy();
    test_strcpy();

    test_memalign();
    #[cfg(not(any(target_os = "macos", target_os = "illumos")))]
    test_reallocarray();

    #[cfg(target_os = "linux")]
    test_sigrt();
}
