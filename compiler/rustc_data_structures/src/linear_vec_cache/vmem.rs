use std::io;
use std::sync::OnceLock;

cfg_select! {
    windows => {
        mod imp {
            use std::ffi::c_void;
            use std::io;

            use windows::Win32::System::Memory::{
                MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
                VirtualFree,
            };
            use windows::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

            pub(super) fn page_size() -> usize {
                let mut info = SYSTEM_INFO::default();
                unsafe {
                    GetSystemInfo(&mut info);
                }
                info.dwPageSize as usize
            }

            pub(super) fn reserve(len: usize) -> io::Result<*mut u8> {
                let ptr = unsafe { VirtualAlloc(None, len, MEM_RESERVE, PAGE_NOACCESS) };
                if ptr.is_null() {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(ptr.cast::<u8>())
                }
            }

            pub(super) fn commit(addr: *mut u8, len: usize) -> io::Result<()> {
                let ptr = unsafe {
                    VirtualAlloc(Some(addr.cast::<c_void>()), len, MEM_COMMIT, PAGE_READWRITE)
                };
                if ptr.is_null() {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(())
                }
            }

            /// SAFETY: `base` must be the exact base pointer previously returned by `reserve` and
            /// must not have been released already.
            pub(super) unsafe fn release(base: *mut u8, _len: usize) -> io::Result<()> {
                unsafe { VirtualFree(base.cast::<c_void>(), 0, MEM_RELEASE) }
                    .map_err(|_| io::Error::last_os_error())
            }
        }
    }
    unix => {
        mod imp {
            use std::io;
            use std::ptr;

            #[cfg(any(target_os = "linux", target_os = "android"))]
            const MAP_ANON_FLAG: i32 = libc::MAP_ANONYMOUS;
            #[cfg(not(any(target_os = "linux", target_os = "android")))]
            const MAP_ANON_FLAG: i32 = libc::MAP_ANON;

            pub(super) fn page_size() -> usize {
                let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
                assert!(page_size > 0);
                page_size as usize
            }

            pub(super) fn reserve(len: usize) -> io::Result<*mut u8> {
                let ptr = unsafe {
                    libc::mmap(
                        ptr::null_mut(),
                        len,
                        libc::PROT_NONE,
                        libc::MAP_PRIVATE | MAP_ANON_FLAG,
                        -1,
                        0,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(ptr.cast::<u8>())
                }
            }

            pub(super) fn commit(addr: *mut u8, len: usize) -> io::Result<()> {
                let status = unsafe {
                    libc::mprotect(
                        addr.cast::<libc::c_void>(),
                        len,
                        libc::PROT_READ | libc::PROT_WRITE,
                    )
                };
                if status == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
            }

            /// SAFETY: `base..base.add(len)` must describe the same mapping previously returned by
            /// `reserve`, and that mapping must not have been released already.
            pub(super) unsafe fn release(base: *mut u8, len: usize) -> io::Result<()> {
                let status = unsafe { libc::munmap(base.cast::<libc::c_void>(), len) };
                if status == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
            }
        }
    }
    _ => {
        mod imp {
            use std::io;

            pub(super) fn page_size() -> usize {
                4096
            }

            pub(super) fn reserve(_len: usize) -> io::Result<*mut u8> {
                Err(io::Error::new(io::ErrorKind::Unsupported, "Linear virtual memory is unsupported on this target"))
            }

            pub(super) fn commit(_addr: *mut u8, _len: usize) -> io::Result<()> {
                Err(io::Error::new(io::ErrorKind::Unsupported, "Linear virtual memory is unsupported on this target"))
            }

            /// SAFETY: This is a no-op fallback; callers must still pass the mapping originally
            /// returned by `reserve` to preserve the contract across platforms.
            pub(super) unsafe fn release(_base: *mut u8, _len: usize) -> io::Result<()> {
                Ok(())
            }
        }
    }
}

pub(super) fn page_size() -> usize {
    static PAGE_SIZE: OnceLock<usize> = OnceLock::new();
    *PAGE_SIZE.get_or_init(imp::page_size)
}

pub(super) fn reserve(len: usize) -> io::Result<*mut u8> {
    imp::reserve(len)
}

pub(super) fn commit(addr: *mut u8, len: usize) -> io::Result<()> {
    imp::commit(addr, len)
}

/// SAFETY: `base` and `len` must satisfy the platform-specific `imp::release` contract, matching
/// a mapping previously returned by `reserve` and not yet released.
pub(super) unsafe fn release(base: *mut u8, len: usize) -> io::Result<()> {
    unsafe { imp::release(base, len) }
}
