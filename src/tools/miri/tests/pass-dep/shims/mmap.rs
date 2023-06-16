//@ignore-target-windows: No libc on Windows
//@compile-flags: -Zmiri-disable-isolation -Zmiri-permissive-provenance
#![feature(strict_provenance)]

use std::{ptr, slice};

fn test_mmap() {
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
    assert!(!ptr.is_null());

    // Ensure that freshly mapped allocations are zeroed
    let slice = unsafe { slice::from_raw_parts_mut(ptr as *mut u8, page_size) };
    assert!(slice.iter().all(|b| *b == 0));

    // Do some writes, make sure they worked
    for b in slice.iter_mut() {
        *b = 1;
    }
    assert!(slice.iter().all(|b| *b == 1));

    // Ensure that we can munmap with just an integer
    let just_an_address = ptr::invalid_mut(ptr.addr());
    let res = unsafe { libc::munmap(just_an_address, page_size) };
    assert_eq!(res, 0i32);
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
}

fn main() {
    test_mmap();
    #[cfg(target_os = "linux")]
    test_mremap();
}
