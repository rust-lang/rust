#![allow(nonstandard_style)]
use std::ffi::c_void;
use std::os::windows::ffi::OsStrExt;
use std::os::windows::raw::HANDLE;
use std::path::Path;
use std::ptr::null_mut as null;
use std::{mem, ptr};

pub fn delete_with_info(path: &Path) -> Result<(), NTSTATUS> {
    let orig_path = path;
    unsafe {
        let path = std::path::absolute(path).unwrap();
        let path: Vec<u16> = r"\??\".encode_utf16().chain(path.as_os_str().encode_wide()).collect();
        let s = UNICODE_STRING {
            Length: (path.len() * 2) as _,
            MaximumLength: (path.len() * 2) as _,
            Buffer: path.as_ptr().cast_mut(),
        };
        let obj_attrs = OBJECT_ATTRIBUTES {
            Length: mem::size_of::<OBJECT_ATTRIBUTES>() as _,
            ObjectName: &s,
            ..mem::zeroed()
        };
        let mut io_status = mem::zeroed();
        let mut handle = null();
        let status = NtOpenFile(
            &mut handle,
            DELETE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
            &obj_attrs,
            &mut io_status,
            1 | 2 | 4,
            FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT,
        );
        if status == STATUS_OBJECT_NAME_NOT_FOUND {
            return Ok(());
        } else if status < 0 {
            panic!("status={:X} path={}", status, orig_path.display());
        }
        io_status = mem::zeroed();
        let mut info: FILE_STAT_INFORMATION = mem::zeroed();
        let status = NtQueryInformationFile(
            handle,
            &mut io_status,
            ptr::from_mut(&mut info).cast(),
            mem::size_of::<FILE_STAT_INFORMATION>() as _,
            FileStatInformation,
        );
        if status < 0 {
            CloseHandle(handle);
            panic!("status={:X}", status);
        }

        let disposition = FILE_DISPOSITION_INFORMATION_EX {
            Flags: FILE_DISPOSITION_DELETE
                | FILE_DISPOSITION_POSIX_SEMANTICS
                | FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE,
        };
        io_status = mem::zeroed();
        let status = NtSetInformationFile(
            handle,
            &mut io_status,
            ptr::from_ref(&disposition).cast(),
            mem::size_of::<FILE_DISPOSITION_INFORMATION_EX>() as _,
            FileDispositionInformationEx,
        );
        CloseHandle(handle);
        println!("{info:#?}");
        if status < 0 {
            panic!("status={:X}", status);
        }
    }
    Ok(())
}

pub fn last_nt_status() -> NTSTATUS {
    unsafe { RtlGetLastNtStatus() }
}

#[repr(C)]
pub struct UNICODE_STRING {
    pub Length: u16,
    pub MaximumLength: u16,
    pub Buffer: *mut u16,
}
#[repr(C)]
struct OBJECT_ATTRIBUTES {
    Length: u32,
    RootDirectory: HANDLE,
    ObjectName: *const UNICODE_STRING,
    Attributes: u32,
    SecurityDescriptor: *const c_void,
    SecurityQualityOfService: *const c_void,
}
#[repr(C)]
pub struct IO_STATUS_BLOCK {
    Status: *mut (),
    Information: usize,
}
#[derive(Debug)]
#[repr(C)]
pub struct FILE_STAT_INFORMATION {
    FileId: i64,
    CreationTime: i64,
    LastAccessTime: i64,
    LastWriteTime: i64,
    ChangeTime: i64,
    AllocationSize: i64,
    EndOfFile: i64,
    FileAttributes: u32,
    ReparseTag: u32,
    NumberOfLinks: u32,
    EffectiveAccess: u32,
}
#[repr(C)]
struct FILE_DISPOSITION_INFORMATION_EX {
    Flags: u32,
}
type NTSTATUS = i32;
const DELETE: u32 = 65536;
const SYNCHRONIZE: u32 = 1048576;
const FILE_READ_ATTRIBUTES: u32 = 128;
const FILE_SYNCHRONOUS_IO_NONALERT: u32 = 0x20;
const FILE_OPEN_REPARSE_POINT: u32 = 0x200000;
const FileStatInformation: i32 = 68;
const FileDispositionInformationEx: i32 = 64;
const FILE_DISPOSITION_DELETE: u32 = 1;
const FILE_DISPOSITION_POSIX_SEMANTICS: u32 = 2;
const FILE_DISPOSITION_IGNORE_READONLY_ATTRIBUTE: u32 = 0x10;
pub const STATUS_OBJECT_NAME_NOT_FOUND: NTSTATUS = 0xC0000034_u32 as _;
//const STATUS_OBJECT_NAME_INVALID: NTSTATUS = 0xC0000033_u32 as _;

#[link(name = "ntdll", kind = "raw-dylib")]
extern "system" {
    fn NtOpenFile(
        FileHandle: *mut HANDLE,
        DesiredAccess: u32,
        ObjectAttributes: *const OBJECT_ATTRIBUTES,
        IoStatusBlock: *mut IO_STATUS_BLOCK,
        ShareAccess: u32,
        OpenOptions: u32,
    ) -> NTSTATUS;
    fn NtQueryInformationFile(
        FileHandle: HANDLE,
        IoStatusBlock: *mut IO_STATUS_BLOCK,
        FileInformation: *mut (),
        Length: u32,
        FileInformationClass: i32,
    ) -> NTSTATUS;
    fn NtSetInformationFile(
        FileHandle: HANDLE,
        IoStatusBlock: *mut IO_STATUS_BLOCK,
        FileInformation: *const (),
        Length: u32,
        FileInformationClass: i32,
    ) -> NTSTATUS;
    fn RtlGetLastNtStatus() -> NTSTATUS;
}
#[link(name = "kernel32", kind = "raw-dylib")]
extern "system" {
    fn CloseHandle(hObject: HANDLE) -> i32;
}
