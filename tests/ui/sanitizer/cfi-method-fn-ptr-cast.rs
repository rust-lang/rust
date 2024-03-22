// Verifies that casting a method to a function pointer works.
//
// FIXME(#122848): Remove only-linux when fixed.
//@ only-linux
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Copt-level=0 -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi
//@ run-pass

trait Trait1 {
    fn foo(&self);
}

struct Type1;

impl Trait1 for Type1 {
    fn foo(&self) {}
}

fn main() {
    let type1 = Type1 {};
    let f = <Type1 as Trait1>::foo;
    f(&type1);
}
