//! Experimental windows hacks to try find what the hecc is holding on to the files that cannot be
//! deleted.

// Adapted from <https://stackoverflow.com/questions/67187979/how-to-call-ntopenfile> from
// Delphi for Rust :3
// Also references <https://gist.github.com/antonioCoco/9db236d6089b4b492746f7de31b21d9d>.

// SAFETY:
// YOLO.

// Windows API naming
#![allow(nonstandard_style)]
// Well because CI does deny-warnings :)
#![deny(unused_imports)]

use std::mem;
use std::os::windows::ffi::OsStrExt;
use std::path::Path;

use anyhow::Result;
use windows::core::PWSTR;
use windows::Wdk::Foundation::OBJECT_ATTRIBUTES;
use windows::Wdk::Storage::FileSystem::{
    NtOpenFile, NtQueryInformationFile, FILE_OPEN_REPARSE_POINT,
};
use windows::Wdk::System::SystemServices::FILE_PROCESS_IDS_USING_FILE_INFORMATION;
use windows::Win32::Foundation::{
    CloseHandle, HANDLE, STATUS_INFO_LENGTH_MISMATCH, UNICODE_STRING,
};
use windows::Win32::Storage::FileSystem::{
    FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
};
use windows::Win32::System::WindowsProgramming::FILE_INFORMATION_CLASS;
use windows::Win32::System::IO::IO_STATUS_BLOCK;

/// Wraps a windows API that returns [`NTSTATUS`]:
///
/// - First convert [`NTSTATUS`] to [`HRESULT`].
/// - Then convert [`HRESULT`] into a [`WinError`] with or without optional info.
macro_rules! try_syscall {
    ($syscall: expr) => {{
        let status = $syscall;
        if status.is_err() {
            ::anyhow::Result::Err(::windows::core::Error::from(status.to_hresult()))?;
        }
    }};
    ($syscall: expr, $additional_info: expr) => {{
        let status = $syscall;
        if status.is_err() {
            ::anyhow::Result::Err(::windows::core::Error::new(
                $syscall.into(),
                $additional_info.into(),
            ))?;
        }
    }};
}

pub(crate) fn process_ids_using_file(path: &Path) -> Result<Vec<usize>> {
    // Gotta have it in UTF-16LE.
    let mut nt_path = {
        let path = std::path::absolute(path)?;
        r"\??\".encode_utf16().chain(path.as_os_str().encode_wide()).collect::<Vec<u16>>()
    };

    let nt_path_unicode_string = UNICODE_STRING {
        Length: u16::try_from(nt_path.len() * 2)?,
        MaximumLength: u16::try_from(nt_path.len() * 2)?,
        Buffer: PWSTR::from_raw(nt_path.as_mut_ptr()),
    };

    let object_attributes = OBJECT_ATTRIBUTES {
        Length: mem::size_of::<OBJECT_ATTRIBUTES>() as _,
        ObjectName: &nt_path_unicode_string,
        ..Default::default()
    };

    let mut io_status = IO_STATUS_BLOCK::default();
    let mut handle = HANDLE::default();

    // https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntopenfile
    try_syscall!(
        unsafe {
            NtOpenFile(
                &mut handle as *mut _,
                FILE_READ_ATTRIBUTES.0,
                &object_attributes,
                &mut io_status as *mut _,
                (FILE_SHARE_READ | FILE_SHARE_DELETE | FILE_SHARE_WRITE).0,
                FILE_OPEN_REPARSE_POINT.0,
            )
        },
        "tried to open file"
    );

    /// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdm/ne-wdm-_file_information_class
    // Remark: apparently windows 0.52 doesn't have this or something, it appears in at least >=
    // 0.53.
    const FileProcessIdsUsingFileInformation: FILE_INFORMATION_CLASS = FILE_INFORMATION_CLASS(47);

    // https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntqueryinformationfile
    const INCREMENT: usize = 8;
    let mut buf = vec![FILE_PROCESS_IDS_USING_FILE_INFORMATION::default(); INCREMENT as usize];
    let mut buf_idx = 0;
    let mut status = unsafe {
        NtQueryInformationFile(
            handle,
            &mut io_status as *mut _,
            buf.as_mut_ptr().cast(),
            (INCREMENT * mem::size_of::<FILE_PROCESS_IDS_USING_FILE_INFORMATION>()) as u32,
            FileProcessIdsUsingFileInformation,
        )
    };
    while status == STATUS_INFO_LENGTH_MISMATCH {
        buf.resize(buf.len() + INCREMENT, FILE_PROCESS_IDS_USING_FILE_INFORMATION::default());
        buf_idx += INCREMENT;
        status = unsafe {
            NtQueryInformationFile(
                handle,
                &mut io_status as *mut _,
                buf.as_mut_ptr()
                    .offset(
                        (buf_idx * mem::size_of::<FILE_PROCESS_IDS_USING_FILE_INFORMATION>())
                            as isize,
                    )
                    .cast(),
                (INCREMENT * mem::size_of::<FILE_PROCESS_IDS_USING_FILE_INFORMATION>()) as u32,
                FileProcessIdsUsingFileInformation,
            )
        };
    }

    let mut process_ids = vec![];

    for FILE_PROCESS_IDS_USING_FILE_INFORMATION {
        NumberOfProcessIdsInList,
        ProcessIdList: [ptr],
    } in buf
    {
        if NumberOfProcessIdsInList >= 1 {
            // only fetch the first one
            process_ids.push(unsafe {
                // This is almost certaintly UB, provenance be damned
                let ptr = ptr as *mut usize;
                *ptr
            });
        }
    }

    try_syscall!(unsafe { CloseHandle(handle) }, "close file handle");

    Ok(process_ids)
}
