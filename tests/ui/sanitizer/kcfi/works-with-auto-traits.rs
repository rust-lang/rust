// Verifies that trait object methods can be called on trait objects with
// additional auto traits.
//
//@ needs-sanitizer-kcfi
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Cpanic=abort -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=kcfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

trait Trait1 {
    fn foo(&self) -> i32;
}

struct Type1;

impl Trait1 for Type1 {
    // <Type1 as Trait1>::foo is transformed into <dyn Trait1 as Trait1>::foo
    fn foo(&self) -> i32 {
        1
    }
}

fn main() {
    let x: &(dyn Trait1 + Send) = &Type1;
    // <dyn Trait1 + Send as Trait1>::foo is transformed into <dyn Trait1 as Trait1>::foo
    assert_eq!(x.foo(), 1);
    let x: &(dyn Trait1 + Send + Sync) = &Type1;
    // <dyn Trait1 + Send + Sync as Trait1>::foo is transformed into <dyn Trait1 as Trait1>::foo
    assert_eq!(x.foo(), 1);
}
