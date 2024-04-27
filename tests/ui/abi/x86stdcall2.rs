//@ run-pass
//@ only-windows

#![allow(non_camel_case_types)]
pub type HANDLE = usize;
pub type DWORD = u32;
pub type SIZE_T = u32;
pub type LPVOID = usize;
pub type BOOL = u8;

mod kernel32 {
    use super::{BOOL, DWORD, HANDLE, LPVOID, SIZE_T};

    extern "system" {
        pub fn GetProcessHeap() -> HANDLE;
        pub fn HeapAlloc(hHeap: HANDLE, dwFlags: DWORD, dwBytes: SIZE_T) -> LPVOID;
        pub fn HeapFree(hHeap: HANDLE, dwFlags: DWORD, lpMem: LPVOID) -> BOOL;
    }
}

pub fn main() {
    let heap = unsafe { kernel32::GetProcessHeap() };
    let mem = unsafe { kernel32::HeapAlloc(heap, 0, 100) };
    assert!(mem != 0);
    let res = unsafe { kernel32::HeapFree(heap, 0, mem) };
    assert!(res != 0);
}
