// Test that println! to a closed stdout does not panic.
// On Windows, close via SetStdHandle to 0.
//@ run-pass

#![feature(rustc_private)]

#[cfg(windows)]
fn close_stdout() {
    type DWORD = u32;
    type HANDLE = *mut u8;
    type BOOL = i32;

    extern "system" {
        fn SetStdHandle(nStdHandle: DWORD, nHandle: HANDLE) -> BOOL;
    }

    const STD_OUTPUT_HANDLE: DWORD = -11i32 as DWORD;
    unsafe { SetStdHandle(STD_OUTPUT_HANDLE, 0 as HANDLE); }
}

#[cfg(not(windows))]
fn close_stdout() {}

fn main() {
    close_stdout();
    println!("hello");
    println!("world");
}
