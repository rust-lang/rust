//@ignore-target: windows # No libc IO on Windows
//@compile-flags: -Zmiri-disable-isolation

fn main() {
    test_file_open_missing_needed_mode();
}

fn test_file_open_missing_needed_mode() {
    let name = b"missing_arg.txt\0";
    let name_ptr = name.as_ptr().cast::<libc::c_char>();
    let _fd = unsafe { libc::open(name_ptr, libc::O_CREAT) }; //~ ERROR: Undefined Behavior: not enough variadic arguments
}
