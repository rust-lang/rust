// Verifies that functions with lifetimes and higher-ranked trait bounds as
// argument types can be called through function pointers and trait objects.
//
//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ ignore-backends: gcc
//@ compile-flags: -Ctarget-feature=-crt-static -Ccodegen-units=1 -Clto -Cprefer-dynamic=off -Copt-level=0 -Zsanitizer=cfi -Cunsafe-allow-abi-mismatch=sanitizer
//@ run-pass

fn bar1(_: &i32) {}

fn bar2(_: &i32, _: &i32) {}

fn bar3(_: &dyn for<'b> Fn(&'b i32)) {}

fn foo1(f: &dyn for<'a> Fn(&'a i32)) -> i32 {
    f(&1);
    1
}

fn foo2(f: for<'a> fn(&'a i32)) -> i32 {
    f(&2);
    2
}

fn foo3(f: for<'a, 'b> fn(&'a i32, &'b i32)) -> i32 {
    f(&3, &4);
    3
}

// A higher-ranked trait bound nested in a higher-ranked function pointer type
fn foo4(f: for<'a> fn(&'a dyn for<'b> Fn(&'b i32))) -> i32 {
    f(&|_x: &i32| {});
    4
}

fn main() {
    let f: fn(&dyn for<'a> Fn(&'a i32)) -> i32 = std::hint::black_box(foo1);
    assert_eq!(f(&|_x: &i32| {}), 1);
    let f: fn(for<'a> fn(&'a i32)) -> i32 = std::hint::black_box(foo2);
    assert_eq!(f(bar1), 2);
    let f: fn(for<'a, 'b> fn(&'a i32, &'b i32)) -> i32 = std::hint::black_box(foo3);
    assert_eq!(f(bar2), 3);
    let f: fn(for<'a> fn(&'a dyn for<'b> Fn(&'b i32))) -> i32 = std::hint::black_box(foo4);
    assert_eq!(f(bar3), 4);
}
