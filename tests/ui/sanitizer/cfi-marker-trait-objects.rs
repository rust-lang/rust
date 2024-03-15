// Test that we can promote closures / fns to trait objects, and call them despite a marker trait.

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C codegen-units=1 -C opt-level=0
//@ run-pass


fn foo() {}

static FOO: &'static (dyn Fn() + Sync) = &foo;
static BAR: &(dyn Fn() -> i32 + Sync) = &|| 3;

fn main() {
    FOO();
    BAR();
}
