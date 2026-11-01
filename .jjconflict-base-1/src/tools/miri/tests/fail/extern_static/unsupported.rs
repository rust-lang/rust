//! Even referencing an unknown `extern static` already triggers an error.

extern "C" {
    static mut FOO: i32;
}

fn main() {
    let _val = std::ptr::addr_of!(FOO); //~ ERROR: is not supported by Miri
}
