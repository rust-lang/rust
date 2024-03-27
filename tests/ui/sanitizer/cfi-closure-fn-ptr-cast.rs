// Verifies that casting a closure to a function pointer works.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0
//@ run-pass

fn main() {
    let f: &fn() = &((|| ()) as _);
    f();
}
