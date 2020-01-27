// needs-sanitizer-support
// only-x86_64
//
// compile-flags: -Zsanitizer=address
// run-fail
// error-pattern: ERROR: AddressSanitizer: stack-use-after-scope

static mut P: *mut usize = std::ptr::null_mut();

fn main() {
    unsafe {
        {
            let mut x = 0;
            P = &mut x;
        }
        std::ptr::write_volatile(P, 123);
    }
}
