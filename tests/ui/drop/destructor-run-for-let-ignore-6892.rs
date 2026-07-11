// https://github.com/rust-lang/rust/issues/6892
//@ run-pass
#![allow(dead_code)]
// Ensures that destructors are run for expressions of the form "let _ = e;"
// where `e` is a type which requires a destructor.

struct Foo;
struct Bar { x: isize }
struct Baz(isize);
enum FooBar { _Foo(Foo), _Bar(usize) }

static mut NUM_DROPS: usize = 0;

fn increment_num_drops() {
    unsafe {
        let num_drops = &raw mut NUM_DROPS;
        num_drops.write(num_drops.read() + 1);
    }
}

fn num_drops() -> usize {
    unsafe { (&raw const NUM_DROPS).read() }
}

impl Drop for Foo {
    fn drop(&mut self) {
        increment_num_drops();
    }
}
impl Drop for Bar {
    fn drop(&mut self) {
        increment_num_drops();
    }
}
impl Drop for Baz {
    fn drop(&mut self) {
        increment_num_drops();
    }
}
impl Drop for FooBar {
    fn drop(&mut self) {
        increment_num_drops();
    }
}

fn main() {
    assert_eq!(num_drops(), 0);
    { let _x = Foo; }
    assert_eq!(num_drops(), 1);
    { let _x = Bar { x: 21 }; }
    assert_eq!(num_drops(), 2);
    { let _x = Baz(21); }
    assert_eq!(num_drops(), 3);
    { let _x = FooBar::_Foo(Foo); }
    assert_eq!(num_drops(), 5);
    { let _x = FooBar::_Bar(42); }
    assert_eq!(num_drops(), 6);

    { let _ = Foo; }
    assert_eq!(num_drops(), 7);
    { let _ = Bar { x: 21 }; }
    assert_eq!(num_drops(), 8);
    { let _ = Baz(21); }
    assert_eq!(num_drops(), 9);
    { let _ = FooBar::_Foo(Foo); }
    assert_eq!(num_drops(), 11);
    { let _ = FooBar::_Bar(42); }
    assert_eq!(num_drops(), 12);
}
