type HANDLE = u32;
type DWORD = u32;
type SIZE_T = u32;
type LPVOID = uint;
type BOOL = u8;

#[cfg(target_os = "win32")]
#[abi = "stdcall"]
native mod kernel32 {
       fn GetProcessHeap() -> HANDLE;
       fn HeapAlloc(hHeap: HANDLE, dwFlags: DWORD, dwBytes: SIZE_T) -> LPVOID;
       fn HeapFree(hHeap: HANDLE, dwFlags: DWORD, lpMem: LPVOID) -> BOOL;
}


#[cfg(target_os = "win32")]
fn main() {
   let heap = kernel32::GetProcessHeap();
   let mem = kernel32::HeapAlloc(heap, 0u32, 100u32);
   assert mem != 0u;
   let res = kernel32::HeapFree(heap, 0u32, mem);
   assert res != 0u8;
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "linux")]
#[cfg(target_os = "freebsd")]
fn main() { }
