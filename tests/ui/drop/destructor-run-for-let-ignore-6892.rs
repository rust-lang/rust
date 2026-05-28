// https://github.com/rust-lang/rust/issues/6892
//@ run-pass
#![allow(dead_code)]
// Ensures that destructors are run for expressions of the form "let _ = e;"
// where `e` is a type which requires a destructor.

// FIXME(static_mut_refs): Do not allow `static_mut_refs` lint
#![allow(static_mut_refs)]

struct Foo;
struct Bar { x: isize }
struct Baz(isize);
enum FooBar { _Foo(Foo), _Bar(usize) }

static mut NUM_DROPS: usize = 0;

impl Drop for Foo {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}
impl Drop for Bar {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}
impl Drop for Baz {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}
impl Drop for FooBar {
    fn drop(&mut self) {
        unsafe { NUM_DROPS += 1; }
    }
}

fn main() {
    assert_eq!(unsafe { NUM_DROPS }, 0);
    { let _x = Foo; }
    assert_eq!(unsafe { NUM_DROPS }, 1);
    { let _x = Bar { x: 21 }; }
    assert_eq!(unsafe { NUM_DROPS }, 2);
    { let _x = Baz(21); }
    assert_eq!(unsafe { NUM_DROPS }, 3);
    { let _x = FooBar::_Foo(Foo); }
    assert_eq!(unsafe { NUM_DROPS }, 5);
    { let _x = FooBar::_Bar(42); }
    assert_eq!(unsafe { NUM_DROPS }, 6);

    { let _ = Foo; }
    assert_eq!(unsafe { NUM_DROPS }, 7);
    { let _ = Bar { x: 21 }; }
    assert_eq!(unsafe { NUM_DROPS }, 8);
    { let _ = Baz(21); }
    assert_eq!(unsafe { NUM_DROPS }, 9);
    { let _ = FooBar::_Foo(Foo); }
    assert_eq!(unsafe { NUM_DROPS }, 11);
    { let _ = FooBar::_Bar(42); }
    assert_eq!(unsafe { NUM_DROPS }, 12);
}
