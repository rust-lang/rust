// Check that dropping a trait object without a principal trait succeeds

//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries works
//@ only-linux
//@ compile-flags: -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=cfi
//@ run-pass

struct CustomDrop;

impl Drop for CustomDrop {
    fn drop(&mut self) {}
}

fn main() {
    let _ = Box::new(CustomDrop) as Box<dyn Send>;
}
