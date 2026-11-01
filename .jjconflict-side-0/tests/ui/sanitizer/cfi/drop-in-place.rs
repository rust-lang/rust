// Verifies that drops can be called on arbitrary trait objects.
//
// FIXME(#122848): Remove only-linux when fixed.
//@ only-linux
//@ ignore-backends: gcc
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Copt-level=0 -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer
//@ run-pass

struct EmptyDrop;

struct NonEmptyDrop;

impl Drop for NonEmptyDrop {
    fn drop(&mut self) {}
}

fn main() {
    let _ = Box::new(EmptyDrop) as Box<dyn Send>;
    let _ = Box::new(NonEmptyDrop) as Box<dyn Send>;
}
