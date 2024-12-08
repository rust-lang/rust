//@ignore-target: windows # No mkstemp on Windows
//@compile-flags: -Zmiri-disable-isolation

fn main() {
    test_mkstemp_immutable_arg();
}

fn test_mkstemp_immutable_arg() {
    let s: *mut libc::c_char = b"fooXXXXXX\0" as *const _ as *mut _;
    let _fd = unsafe { libc::mkstemp(s) }; //~ ERROR: Undefined Behavior: writing to alloc1 which is read-only
}
