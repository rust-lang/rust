//! Disable write cache flushing (i.e. write-durability guarantees) on Windows.
//! Only run this on ephemeral machines, otherwise it may cause data loss.
use std::ffi::CString;
use std::ptr::{null, null_mut};
use std::{mem, ptr};

use windows_sys::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileA, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
};
use windows_sys::Win32::System::IO::DeviceIoControl;

const IOCTL_DISK_GET_CACHE_SETTINGS: u32 = 0x740e0;
const IOCTL_DISK_SET_CACHE_SETTINGS: u32 = 0x7c0e4;

#[repr(C)]
struct DiskCacheSettings {
    pub version: u32,
    pub state: u32,
    pub is_power_protected: u8,
}

fn main() {
    // 3 drives ought to be enough for everyone
    for i in 0..=2 {
        unsafe {
            let disk = format!(r#"\\.\PHYSICALDRIVE{}"#, i);
            let hd1 = CString::new(disk.clone()).unwrap();
            let handle = CreateFileA(
                hd1.as_ptr().cast(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                null(),
                OPEN_EXISTING,
                0,
                0,
            );

            if handle == INVALID_HANDLE_VALUE {
                let error_code = windows_sys::Win32::Foundation::GetLastError();
                eprintln!("Failed to open device {disk}, {:x}", error_code);
                continue;
            }

            let mut settings: DiskCacheSettings = mem::zeroed();
            let mut lpbytesreturned: u32 = 0;

            let r = DeviceIoControl(
                handle,
                IOCTL_DISK_GET_CACHE_SETTINGS,
                null(),
                0,
                ptr::from_mut(&mut settings).cast(),
                size_of::<DiskCacheSettings>() as u32,
                &mut lpbytesreturned,
                null_mut(),
            );
            if r == 0 {
                eprintln!("Failed to get cache settings");
                continue;
            }

            // Enable YOLO mode
            settings.is_power_protected = 1;

            let r = DeviceIoControl(
                handle,
                IOCTL_DISK_SET_CACHE_SETTINGS,
                ptr::from_ref(&settings).cast(),
                size_of::<DiskCacheSettings>() as u32,
                null_mut(),
                0,
                &mut lpbytesreturned,
                null_mut(),
            );
            if r == 0 {
                eprintln!("Failed to set cache settings");
                continue;
            }

            // verify that it sticks

            let r = DeviceIoControl(
                handle,
                IOCTL_DISK_GET_CACHE_SETTINGS,
                null(),
                0,
                ptr::from_mut(&mut settings).cast(),
                size_of::<DiskCacheSettings>() as u32,
                &mut lpbytesreturned,
                null_mut(),
            );

            if r == 0 {
                eprintln!("Failed to get cache settings the second time");
                continue;
            }

            if settings.is_power_protected != 1 {
                eprintln!(
                    "Failed to enable YOLO mode for drive {i}, is_power_protected was not retained"
                );
            } else {
                eprintln!("YOLO mode enabled successfully for drive {i}!");
            }
        }
    }
}
