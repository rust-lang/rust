// Tests that converting a closure to a function pointer works

//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags:-C codegen-units=1 -C opt-level=0
//@ run-pass

pub fn main() {
    let f: &fn() = &((|| ()) as _);
    f();
}
