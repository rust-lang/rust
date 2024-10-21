//@ignore-target: windows # No mmap on Windows
//@compile-flags: -Zmiri-disable-isolation -Zmiri-permissive-provenance

use std::io::Error;
use std::{ptr, slice};

fn test_mmap<Offset: Default>(
    mmap: unsafe extern "C" fn(
        *mut libc::c_void,
        libc::size_t,
        libc::c_int,
        libc::c_int,
        libc::c_int,
        Offset,
    ) -> *mut libc::c_void,
) {
    let page_size = page_size::get();
    let ptr = unsafe {
        mmap(
            ptr::null_mut(),
            page_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            Default::default(),
        )
    };
    assert!(!ptr.is_null());

    // Ensure that freshly mapped allocations are zeroed
    let slice = unsafe { slice::from_raw_parts_mut(ptr as *mut u8, page_size) };
    assert!(slice.iter().all(|b| *b == 0));

    // Do some writes, make sure they worked
    for b in slice.iter_mut() {
        *b = 1;
    }
    assert!(slice.iter().all(|b| *b == 1));

    // Ensure that we can munmap
    let res = unsafe { libc::munmap(ptr, page_size) };
    assert_eq!(res, 0i32);

    // Test all of our error conditions
    let ptr = unsafe {
        mmap(
            ptr::null_mut(),
            page_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_SHARED, // Can't be both private and shared
            -1,
            Default::default(),
        )
    };
    assert_eq!(ptr, libc::MAP_FAILED);
    assert_eq!(Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);

    let ptr = unsafe {
        mmap(
            ptr::null_mut(),
            0, // Can't map no memory
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            Default::default(),
        )
    };
    assert_eq!(ptr, libc::MAP_FAILED);
    assert_eq!(Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);

    // We report an error for mappings whose length cannot be rounded up to a multiple of
    // the page size.
    let ptr = unsafe {
        mmap(
            ptr::null_mut(),
            usize::MAX - 1,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            Default::default(),
        )
    };
    assert_eq!(ptr, libc::MAP_FAILED);

    // We report an error when trying to munmap an address which is not a multiple of the page size
    let res = unsafe { libc::munmap(ptr::without_provenance_mut(1), page_size) };
    assert_eq!(res, -1);
    assert_eq!(Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);

    // We report an error when trying to munmap a length that cannot be rounded up to a multiple of
    // the page size.
    let res = unsafe { libc::munmap(ptr::without_provenance_mut(page_size), usize::MAX - 1) };
    assert_eq!(res, -1);
    assert_eq!(Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);
}

#[cfg(target_os = "linux")]
fn test_mremap() {
    let page_size = page_size::get();
    let ptr = unsafe {
        libc::mmap(
            ptr::null_mut(),
            page_size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    let slice = unsafe { slice::from_raw_parts_mut(ptr as *mut u8, page_size) };
    for b in slice.iter_mut() {
        *b = 1;
    }

    let ptr = unsafe { libc::mremap(ptr, page_size, page_size * 2, libc::MREMAP_MAYMOVE) };
    assert!(!ptr.is_null());

    let slice = unsafe { slice::from_raw_parts_mut(ptr as *mut u8, page_size * 2) };
    assert!(&slice[..page_size].iter().all(|b| *b == 1));
    assert!(&slice[page_size..].iter().all(|b| *b == 0));

    let res = unsafe { libc::munmap(ptr, page_size * 2) };
    assert_eq!(res, 0i32);

    // Test all of our error conditions
    // Not aligned
    let ptr = unsafe {
        libc::mremap(ptr::without_provenance_mut(1), page_size, page_size, libc::MREMAP_MAYMOVE)
    };
    assert_eq!(ptr, libc::MAP_FAILED);
    assert_eq!(Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);

    // Zero size
    let ptr = unsafe { libc::mremap(ptr::null_mut(), page_size, 0, libc::MREMAP_MAYMOVE) };
    assert_eq!(ptr, libc::MAP_FAILED);
    assert_eq!(Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);

    // Not setting MREMAP_MAYMOVE
    let ptr = unsafe { libc::mremap(ptr::null_mut(), page_size, page_size, 0) };
    assert_eq!(ptr, libc::MAP_FAILED);
    assert_eq!(Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);
}

fn main() {
    test_mmap(libc::mmap);
    #[cfg(target_os = "linux")]
    test_mmap(libc::mmap64);
    #[cfg(target_os = "linux")]
    test_mremap();
}
