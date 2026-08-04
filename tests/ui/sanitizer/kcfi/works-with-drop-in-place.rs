// Verifies that drops can be called on arbitrary trait objects, including trait
// objects without a principal trait.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

// A type without a Drop implementation
struct Type1;

// A type with a Drop implementation
struct Type2;

impl Drop for Type2 {
    fn drop(&mut self) {}
}

fn main() {
    // Dropping the values below calls the drop glue of their types through the vtable of the
    // dyn Send trait object (i.e., a trait object without a principal trait). Both the drop
    // glue and the virtual drop calls to it are transformed into drop_in_place::<dyn Drop>.
    let _ = Box::new(Type1) as Box<dyn Send>;
    let _ = Box::new(Type2) as Box<dyn Send>;
}
