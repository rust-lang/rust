#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// This test verifies that a `&pin mut Foo` can be projected to a pinned
// reference `&pin mut T` of a `?Unpin` field marked by `#[pin]`, and can
// be projected to an unpinned reference `&mut U` of an `Unpin` field not
// marked by `#[pin]`.

struct Foo<T, U: Unpin> {
    #[pin]
    x: T,
    y: U,
}

struct Bar<T, U: Unpin>(#[pin] T, U);

enum Baz<T, U: Unpin> {
    Foo(#[pin] T, U),
    Bar {
        #[pin]
        x: T,
        y: U,
    },
}

trait IsPinMut {}
trait IsPinConst {}
trait IsRefMut {}
trait IsRef {}
impl<T: ?Sized> IsPinMut for &pin mut T {}
impl<T: ?Sized> IsPinConst for &pin const T {}
impl<T: ?Sized> IsRefMut for &mut T {}
impl<T: ?Sized> IsRef for &T {}

fn assert_pin_mut<T: IsPinMut>(_: T) {}
fn assert_pin_const<T: IsPinConst>(_: T) {}
fn assert_ref_mut<T: IsRefMut>(_: T) {}
fn assert_ref<T: IsRef>(_: T) {}

fn foo_mut<T, U: Unpin>(foo: &pin mut Foo<T, U>) {
    let Foo { x, y } = foo;
    assert_pin_mut(x);
    assert_ref_mut(y);
}

fn foo_const<T, U: Unpin>(foo: &pin const Foo<T, U>) {
    let Foo { x, y } = foo;
    assert_pin_const(x);
    assert_ref(y);
}

fn bar_mut<T, U: Unpin>(bar: &pin mut Bar<T, U>) {
    let Bar(x, y) = bar;
    assert_pin_mut(x);
    assert_ref_mut(y);
}

fn bar_const<T, U: Unpin>(bar: &pin const Bar<T, U>) {
    let Bar(x, y) = bar;
    assert_pin_const(x);
    assert_ref(y);
}

fn baz_mut<T, U: Unpin>(baz: &pin mut Baz<T, U>) {
    match baz {
        Baz::Foo(x, y) => {
            assert_pin_mut(x);
            assert_ref_mut(y);
        }
        Baz::Bar { x, y } => {
            assert_pin_mut(x);
            assert_ref_mut(y);
        }
    }
}

fn baz_const<T, U: Unpin>(baz: &pin const Baz<T, U>) {
    match baz {
        Baz::Foo(x, y) => {
            assert_pin_const(x);
            assert_ref(y);
        }
        Baz::Bar { x, y } => {
            assert_pin_const(x);
            assert_ref(y);
        }
    }
}

fn main() {}
