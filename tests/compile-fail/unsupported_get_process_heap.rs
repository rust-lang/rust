//! `GetProcessHeap()` is special on Windows that it's only supported within libstd.
//! (On Linux and macOS, it's just always unsupported.)

fn main() {
    extern "system" {
        fn GetProcessHeap() -> *mut std::ffi::c_void;
    }
    unsafe {
        GetProcessHeap();
        //~^ ERROR unsupported operation: can't call foreign function: GetProcessHeap
    }
}
