// Check that dropping a trait object without a principal trait succeeds

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C codegen-units=1 -C opt-level=0
//@ run-pass
// Check that trait objects without a principal can be dropped.

struct CustomDrop;
impl Drop for CustomDrop {
    fn drop(&mut self) {}
}

fn main() {
    let _ = Box::new(CustomDrop) as Box<dyn Send>;
}
