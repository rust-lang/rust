// Check that dropping a trait object without a principal trait succeeds

//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries works
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi -C unsafe-allow-abi-mismatch=sanitizer
//@ compile-flags: -C target-feature=-crt-static -C codegen-units=1 -C opt-level=0
// FIXME(#118761) Should be run-pass once the labels on drop are compatible.
// This test is being landed ahead of that to test that the compiler doesn't ICE while labeling the
// callsite for a drop, but the vtable doesn't have the correct label yet.
//@ build-pass

struct CustomDrop;

impl Drop for CustomDrop {
    fn drop(&mut self) {}
}

fn main() {
    let _ = Box::new(CustomDrop) as Box<dyn Send>;
}
