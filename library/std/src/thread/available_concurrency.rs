use crate::io;
use crate::num::NonZeroUsize;

/// Returns the number of hardware threads available to the program.
///
/// This value should be considered only a hint.
///
/// # Platform-specific behavior
///
/// If interpreted as the number of actual hardware threads, it may undercount on
/// Windows systems with more than 64 hardware threads. If interpreted as the
/// available concurrency for that process, it may overcount on Windows systems
/// when limited by a process wide affinity mask or job object limitations, and
/// it may overcount on Linux systems when limited by a process wide affinity
/// mask or affected by cgroups limits.
///
/// # Errors
///
/// This function will return an error in the following situations, but is not
/// limited to just these cases:
///
/// - If the number of hardware threads is not known for the target platform.
/// - The process lacks permissions to view the number of hardware threads
///   available.
///
/// # Examples
///
/// ```
/// # #![allow(dead_code)]
/// #![feature(available_concurrency)]
/// use std::thread;
///
/// let count = thread::available_concurrency().map(|n| n.get()).unwrap_or(1);
/// ```
#[unstable(feature = "available_concurrency", issue = "74479")]
pub fn available_concurrency() -> io::Result<NonZeroUsize> {
    available_concurrency_internal()
}

cfg_if::cfg_if! {
    if #[cfg(windows)] {
        #[allow(nonstandard_style)]
        fn available_concurrency_internal() -> io::Result<NonZeroUsize> {
            #[repr(C)]
            struct SYSTEM_INFO {
                wProcessorArchitecture: u16,
                wReserved: u16,
                dwPageSize: u32,
                lpMinimumApplicationAddress: *mut u8,
                lpMaximumApplicationAddress: *mut u8,
                dwActiveProcessorMask: *mut u8,
                dwNumberOfProcessors: u32,
                dwProcessorType: u32,
                dwAllocationGranularity: u32,
                wProcessorLevel: u16,
                wProcessorRevision: u16,
            }
            extern "system" {
                fn GetSystemInfo(info: *mut SYSTEM_INFO) -> i32;
            }
            let res = unsafe {
                let mut sysinfo = crate::mem::zeroed();
                GetSystemInfo(&mut sysinfo);
                sysinfo.dwNumberOfProcessors as usize
            };
            match res {
                0 => Err(io::Error::new(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform")),
                cpus => Ok(unsafe { NonZeroUsize::new_unchecked(cpus) }),
            }
        }
    } else if #[cfg(any(
        target_os = "android",
        target_os = "emscripten",
        target_os = "fuchsia",
        target_os = "ios",
        target_os = "linux",
        target_os = "macos",
        target_os = "solaris",
        target_os = "illumos",
    ))] {
        fn available_concurrency_internal() -> io::Result<NonZeroUsize> {
            match unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) } {
                -1 => Err(io::Error::last_os_error()),
                0 => Err(io::Error::new(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform")),
                cpus => Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) }),
            }
        }
    } else if #[cfg(any(target_os = "freebsd", target_os = "dragonfly", target_os = "netbsd"))] {
        fn available_concurrency_internal() -> io::Result<NonZeroUsize> {
            use crate::ptr;

            let mut cpus: libc::c_uint = 0;
            let mut cpus_size = crate::mem::size_of_val(&cpus);

            unsafe {
                cpus = libc::sysconf(libc::_SC_NPROCESSORS_ONLN) as libc::c_uint;
            }

            // Fallback approach in case of errors or no hardware threads.
            if cpus < 1 {
                let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];
                let res = unsafe {
                    libc::sysctl(
                        mib.as_mut_ptr(),
                        2,
                        &mut cpus as *mut _ as *mut _,
                        &mut cpus_size as *mut _ as *mut _,
                        ptr::null_mut(),
                        0,
                    )
                };

                // Handle errors if any.
                if res == -1 {
                    return Err(io::Error::last_os_error());
                } else if cpus == 0 {
                    return Err(io::Error::new(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform"));
                }
            }
            Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) })
        }
    } else if #[cfg(target_os = "openbsd")] {
        fn available_concurrency_internal() -> io::Result<NonZeroUsize> {
            use crate::ptr;

            let mut cpus: libc::c_uint = 0;
            let mut cpus_size = crate::mem::size_of_val(&cpus);
            let mut mib = [libc::CTL_HW, libc::HW_NCPU, 0, 0];

            let res = unsafe {
                libc::sysctl(
                    mib.as_mut_ptr(),
                    2,
                    &mut cpus as *mut _ as *mut _,
                    &mut cpus_size as *mut _ as *mut _,
                    ptr::null_mut(),
                    0,
                )
            };

            // Handle errors if any.
            if res == -1 {
                return Err(io::Error::last_os_error());
            } else if cpus == 0 {
                return Err(io::Error::new(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform"));
            }

            Ok(unsafe { NonZeroUsize::new_unchecked(cpus as usize) })
        }
    } else {
        // FIXME: implement on vxWorks, Redox, HermitCore, Haiku, l4re
        fn available_concurrency_internal() -> io::Result<NonZeroUsize> {
            Err(io::Error::new(io::ErrorKind::NotFound, "The number of hardware threads is not known for the target platform"))
        }
    }
}
