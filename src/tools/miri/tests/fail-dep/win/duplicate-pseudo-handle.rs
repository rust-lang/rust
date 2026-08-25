//@only-target: windows # Uses win32 api functions
use windows_sys::Win32::Foundation::{CloseHandle, DUPLICATE_SAME_ACCESS, DuplicateHandle};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThread};

fn main() {
    unsafe {
        let cur_proc = GetCurrentProcess();

        let pseudo = GetCurrentThread();
        let mut out = std::mem::zeroed();
        let res =
            DuplicateHandle(cur_proc, pseudo, cur_proc, &mut out, 0, 0, DUPLICATE_SAME_ACCESS);
        //~^ERROR: pseudo handle
        assert!(res != 0);
        assert!(out.addr() != 0);
        // Since the original handle was a pseudo handle, we must return something different.
        assert!(out != pseudo);
        // And closing it should work (which it does not for a pseudo handle).
        let res = CloseHandle(out);
        assert!(res != 0);
    }
}
