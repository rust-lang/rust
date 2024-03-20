// Verifies that custom drops can be called on arbitraty trait objects.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0
//@ run-pass

struct CustomDrop;
impl Drop for CustomDrop {
    fn drop(&mut self) {}
}

fn main() {
    let _ = Box::new(CustomDrop) as Box<dyn Send>;
}
