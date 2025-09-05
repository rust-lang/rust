//@ needs-sanitizer-support
//@ needs-sanitizer-address
//@ ignore-cross-compile
//
//@ compile-flags: -Zsanitizer=address -C unsafe-allow-abi-mismatch=sanitizer
//@ run-fail-or-crash
//@ error-pattern: ERROR: AddressSanitizer: stack-use-after-scope

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
