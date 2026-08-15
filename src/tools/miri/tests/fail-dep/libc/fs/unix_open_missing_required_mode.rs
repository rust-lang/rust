//@ignore-target: windows # No libc IO on Windows
//@compile-flags: -Zmiri-disable-isolation

fn main() {
    test_file_open_missing_needed_mode();
}

fn test_file_open_missing_needed_mode() {
    let name = c"missing_arg.txt".as_ptr();
    let _fd = unsafe { libc::open(name, libc::O_CREAT) }; //~ ERROR: Undefined Behavior: not enough variadic arguments
}
