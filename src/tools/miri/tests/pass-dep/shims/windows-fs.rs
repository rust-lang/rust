//@only-target: windows # this directly tests windows-only functions
//@compile-flags: -Zmiri-disable-isolation
#![allow(nonstandard_style)]

use std::os::windows::ffi::OsStrExt;
use std::path::Path;
use std::ptr;

#[path = "../../utils/mod.rs"]
mod utils;

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_ACCESS_DENIED, ERROR_ALREADY_EXISTS, ERROR_IO_DEVICE, GENERIC_READ,
    GENERIC_WRITE, GetLastError, RtlNtStatusToDosError, STATUS_ACCESS_DENIED,
    STATUS_IO_DEVICE_ERROR,
};
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, CREATE_ALWAYS, CREATE_NEW, CreateFileW, FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_NORMAL, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
    FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, GetFileInformationByHandle, OPEN_ALWAYS,
    OPEN_EXISTING,
};

fn main() {
    unsafe {
        test_create_dir_file();
        test_create_normal_file();
        test_create_always_twice();
        test_open_always_twice();
        test_open_dir_reparse();
        test_ntstatus_to_dos();
    }
}

unsafe fn test_create_dir_file() {
    let temp = utils::tmp();
    let raw_path = to_wide_cstr(&temp);
    // Open the `temp` directory.
    let handle = CreateFileW(
        raw_path.as_ptr(),
        GENERIC_READ,
        FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        ptr::null_mut(),
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS,
        ptr::null_mut(),
    );
    assert_ne!(handle.addr(), usize::MAX, "CreateFileW Failed: {}", GetLastError());
    let mut info = std::mem::zeroed::<BY_HANDLE_FILE_INFORMATION>();
    if GetFileInformationByHandle(handle, &mut info) == 0 {
        panic!("Failed to get file information")
    };
    assert!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0);
    if CloseHandle(handle) == 0 {
        panic!("Failed to close file")
    };
}

unsafe fn test_create_normal_file() {
    let temp = utils::tmp().join("test.txt");
    let raw_path = to_wide_cstr(&temp);
    let handle = CreateFileW(
        raw_path.as_ptr(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        ptr::null_mut(),
        CREATE_NEW,
        0,
        ptr::null_mut(),
    );
    assert_ne!(handle.addr(), usize::MAX, "CreateFileW Failed: {}", GetLastError());
    let mut info = std::mem::zeroed::<BY_HANDLE_FILE_INFORMATION>();
    if GetFileInformationByHandle(handle, &mut info) == 0 {
        panic!("Failed to get file information: {}", GetLastError())
    };
    assert!(info.dwFileAttributes & FILE_ATTRIBUTE_NORMAL != 0);
    if CloseHandle(handle) == 0 {
        panic!("Failed to close file")
    };

    // Test metadata-only handle
    let handle = CreateFileW(
        raw_path.as_ptr(),
        0,
        FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        ptr::null_mut(),
        OPEN_EXISTING,
        0,
        ptr::null_mut(),
    );
    assert_ne!(handle.addr(), usize::MAX, "CreateFileW Failed: {}", GetLastError());
    let mut info = std::mem::zeroed::<BY_HANDLE_FILE_INFORMATION>();
    if GetFileInformationByHandle(handle, &mut info) == 0 {
        panic!("Failed to get file information: {}", GetLastError())
    };
    assert!(info.dwFileAttributes & FILE_ATTRIBUTE_NORMAL != 0);
    if CloseHandle(handle) == 0 {
        panic!("Failed to close file")
    };
}

/// Tests that CREATE_ALWAYS sets the error value correctly based on whether the file already exists
unsafe fn test_create_always_twice() {
    let temp = utils::tmp().join("test_create_always.txt");
    let raw_path = to_wide_cstr(&temp);
    let handle = CreateFileW(
        raw_path.as_ptr(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        ptr::null_mut(),
        CREATE_ALWAYS,
        0,
        ptr::null_mut(),
    );
    assert_ne!(handle.addr(), usize::MAX, "CreateFileW Failed: {}", GetLastError());
    assert_eq!(GetLastError(), 0);
    if CloseHandle(handle) == 0 {
        panic!("Failed to close file")
    };

    let handle = CreateFileW(
        raw_path.as_ptr(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        ptr::null_mut(),
        CREATE_ALWAYS,
        0,
        ptr::null_mut(),
    );
    assert_ne!(handle.addr(), usize::MAX, "CreateFileW Failed: {}", GetLastError());
    assert_eq!(GetLastError(), ERROR_ALREADY_EXISTS);
    if CloseHandle(handle) == 0 {
        panic!("Failed to close file")
    };
}

/// Tests that OPEN_ALWAYS sets the error value correctly based on whether the file already exists
unsafe fn test_open_always_twice() {
    let temp = utils::tmp().join("test_open_always.txt");
    let raw_path = to_wide_cstr(&temp);
    let handle = CreateFileW(
        raw_path.as_ptr(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        ptr::null_mut(),
        OPEN_ALWAYS,
        0,
        ptr::null_mut(),
    );
    assert_ne!(handle.addr(), usize::MAX, "CreateFileW Failed: {}", GetLastError());
    assert_eq!(GetLastError(), 0);
    if CloseHandle(handle) == 0 {
        panic!("Failed to close file")
    };

    let handle = CreateFileW(
        raw_path.as_ptr(),
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        ptr::null_mut(),
        OPEN_ALWAYS,
        0,
        ptr::null_mut(),
    );
    assert_ne!(handle.addr(), usize::MAX, "CreateFileW Failed: {}", GetLastError());
    assert_eq!(GetLastError(), ERROR_ALREADY_EXISTS);
    if CloseHandle(handle) == 0 {
        panic!("Failed to close file")
    };
}

// TODO: Once we support more of the std API, it would be nice to test against an actual symlink
unsafe fn test_open_dir_reparse() {
    let temp = utils::tmp();
    let raw_path = to_wide_cstr(&temp);
    // Open the `temp` directory.
    let handle = CreateFileW(
        raw_path.as_ptr(),
        GENERIC_READ,
        FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE,
        ptr::null_mut(),
        OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
        ptr::null_mut(),
    );
    assert_ne!(handle.addr(), usize::MAX, "CreateFileW Failed: {}", GetLastError());
    let mut info = std::mem::zeroed::<BY_HANDLE_FILE_INFORMATION>();
    if GetFileInformationByHandle(handle, &mut info) == 0 {
        panic!("Failed to get file information")
    };
    assert!(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0);
    if CloseHandle(handle) == 0 {
        panic!("Failed to close file")
    };
}

unsafe fn test_ntstatus_to_dos() {
    // We won't test all combinations, just a couple common ones
    assert_eq!(RtlNtStatusToDosError(STATUS_IO_DEVICE_ERROR), ERROR_IO_DEVICE);
    assert_eq!(RtlNtStatusToDosError(STATUS_ACCESS_DENIED), ERROR_ACCESS_DENIED);
}

fn to_wide_cstr(path: &Path) -> Vec<u16> {
    let mut raw_path = path.as_os_str().encode_wide().collect::<Vec<_>>();
    raw_path.extend([0, 0]);
    raw_path
}
