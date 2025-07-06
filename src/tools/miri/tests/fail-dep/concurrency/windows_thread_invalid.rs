//! Ensure we error if thread functions are called with invalid handles
//@only-target: windows # testing Windows API

use windows_sys::Win32::System::Threading::GetThreadId;

fn main() {
    let _tid = unsafe { GetThreadId(std::ptr::dangling_mut()) };
    //~^ ERROR: invalid handle
}
