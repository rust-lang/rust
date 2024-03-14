// Check that the type of trait methods on a concrete type are not abstracted

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C codegen-units=1 -C opt-level=0
//@ run-pass

trait Foo {
    fn foo(&self);
}

struct S;

impl Foo for S {
    fn foo(&self) {}
}

struct S2 {
    f: fn(&S)
}

impl S2 {
    fn foo(&self, s: &S) {
        (self.f)(s)
    }
}

fn main() {
    S2 { f: <S as Foo>::foo }.foo(&S)
}
