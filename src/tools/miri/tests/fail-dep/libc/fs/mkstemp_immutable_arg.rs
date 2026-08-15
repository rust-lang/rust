//@ignore-target: windows # No mkstemp on Windows
//@compile-flags: -Zmiri-disable-isolation

fn main() {
    test_mkstemp_immutable_arg();
}

fn test_mkstemp_immutable_arg() {
    let s: *mut libc::c_char = c"fooXXXXXX".as_ptr().cast_mut();
    let _fd = unsafe { libc::mkstemp(s) }; //~ ERROR: Undefined Behavior: writing to alloc1 which is read-only
}
