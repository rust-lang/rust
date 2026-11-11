use crate::ffi::c_void;
use crate::mem::MaybeUninit;
use crate::os::windows::io::{AsHandle, AsRawHandle, BorrowedHandle};
use crate::sys::api::utf16;
use crate::sys::c;

pub fn is_terminal(h: &impl AsHandle) -> bool {
    handle_is_console(h.as_handle())
}

fn handle_is_console(handle: BorrowedHandle<'_>) -> bool {
    // A null handle means the process has no console.
    if handle.as_raw_handle().is_null() {
        return false;
    }

    cfg_select! {
        target_vendor = "win7" => {
            let mut out = 0;
            if unsafe { c::GetConsoleMode(handle.as_raw_handle(), &mut out) != 0 } {
                // False positives aren't possible. If we got a console then we definitely have a console.
                return true;
            }

            // Early return if the handle is not FILE_TYPE_PIPE, implying it cannot be an msys PTY.
            if unsafe { c::GetFileType(handle.as_raw_handle()) != 0x0003 } {
                return false;
            }
        }
        _ => unsafe {
            use crate::mem::MaybeUninit;
            // `GetConsoleMode` incurs a remote RPC round-trip (= bad), so it's best avoided.
            // `NtQueryVolumeInformationFile` lets us test the handle directly.
            // The required `FILE_DEVICE_CONSOLE` type was introduced in Windows 8.
            // Fun fact: `GetFileType` uses `NtQueryVolumeInformationFile` internally.
            let mut io_status = MaybeUninit::<c::IO_STATUS_BLOCK>::uninit();
            let mut device_info = MaybeUninit::<c::FILE_FS_DEVICE_INFORMATION>::uninit();
            let status = c::NtQueryVolumeInformationFile(
                handle.as_raw_handle(),
                io_status.as_mut_ptr(),
                device_info.as_mut_ptr().cast(),
                size_of_val(&device_info) as u32,
                c::FileFsDeviceInformation,
            );
            if status < 0 {
                return false;
            }

            match device_info.assume_init_ref().DeviceType {
                c::FILE_DEVICE_CONSOLE => return true,
                // If it's a named pipe, it may be an msys PTY.
                c::FILE_DEVICE_NAMED_PIPE => {} // aka FILE_TYPE_PIPE
                _ => return false,
            }
        },
    }

    // Otherwise, we fall back to an msys hack to see if we can detect the presence of a pty.
    msys_tty_on(handle)
}

fn msys_tty_on(handle: BorrowedHandle<'_>) -> bool {
    /// Mirrors [`FILE_NAME_INFO`], giving it a fixed length that we can stack
    /// allocate
    ///
    /// [`FILE_NAME_INFO`]: https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_name_info
    #[repr(C)]
    #[allow(non_snake_case)]
    struct FILE_NAME_INFO {
        FileNameLength: u32,
        FileName: [MaybeUninit<u16>; c::MAX_PATH as usize],
    }
    let mut name_info =
        FILE_NAME_INFO { FileNameLength: 0, FileName: [const { MaybeUninit::uninit() }; _] };
    // Safety: buffer length is fixed.
    let res = unsafe {
        c::GetFileInformationByHandleEx(
            handle.as_raw_handle(),
            c::FileNameInfo,
            (&raw mut name_info) as *mut c_void,
            size_of::<FILE_NAME_INFO>() as u32,
        )
    };
    if res == 0 {
        return false;
    }

    // Use `get` because `FileNameLength` can be out of range.
    let s = match name_info.FileName.get(..name_info.FileNameLength as usize / 2) {
        None => return false,
        // SAFETY: As per the API contract, `FileNameLength`-many bytes have been initialized.
        Some(s) => unsafe { s.assume_init_ref() },
    };

    // Get the file name only.
    let name = s.rsplit(|&unit| unit == b'\\' as u16).next().unwrap_or(s);

    // This checks whether 'pty' exists in the file name, which indicates that
    // a pseudo-terminal is attached. To mitigate against false positives
    // (e.g., an actual file name that contains 'pty'), we also require that
    // the file name begins with either the strings 'msys-' or 'cygwin-'.)
    //
    // Lambdas are used to allow the compiler to short-circuit the is_pty check.
    let is_msys = || name.starts_with(utf16!("msys-")) || name.starts_with(utf16!("cygwin-"));
    let is_pty = || name.array_windows::<4>().any(|part| part == utf16!("-pty"));
    is_msys() && is_pty()
}
