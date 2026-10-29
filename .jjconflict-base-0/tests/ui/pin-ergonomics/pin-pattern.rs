//@ edition:2024
#![feature(pin_ergonomics)]
#![allow(incomplete_features)]

// This test verifies:
// - a `&pin mut $pat` can be used to match on a pinned reference type `&pin mut T`;
// - the subpattern can only convert the binding mode `&pin mut` to `&mut` when `T: Unpin`;
// - the subpattern can only remove the binding mode `&pin mut` when `T: Copy`;

#[pin_v2]
struct Foo<T>(T);

trait IsPinMut {}
trait IsPinConst {}
trait IsMut {}
trait IsRef {}
impl<T: ?Sized> IsPinMut for &pin mut T {}
impl<T: ?Sized> IsPinConst for &pin const T {}
impl<T: ?Sized> IsMut for &mut T {}
impl<T: ?Sized> IsRef for &T {}

fn assert_pin_mut<T: IsPinMut>(_: T) {}
fn assert_pin_const<T: IsPinConst>(_: T) {}
fn assert_mut<T: IsMut>(_: T) {}
fn assert_ref<T: IsRef>(_: T) {}
fn assert_ty<T>(_: T) {}

fn normal<T>(foo_mut: &pin mut Foo<T>, foo_const: &pin const Foo<T>) {
    let &pin mut Foo(ref pin mut x) = foo_mut; // ok
    assert_pin_mut(x);

    let &pin const Foo(ref pin const x) = foo_const; // ok
    assert_pin_const(x);
}

fn by_value_copy<T: Copy>(foo_mut: &pin mut Foo<T>, foo_const: &pin const Foo<T>) {
    let &pin mut Foo(x) = foo_mut;
    assert_ty::<T>(x);

    let &pin const Foo(x) = foo_const;
    assert_ty::<T>(x);
}

fn by_value_non_copy<T>(foo_mut: &pin mut Foo<T>, foo_const: &pin const Foo<T>) {
    let &pin mut Foo(x) = foo_mut;
    //~^ ERROR cannot move out of `foo_mut.pointer` which is behind a mutable reference
    assert_ty::<T>(x);

    let &pin const Foo(x) = foo_const;
    //~^ ERROR cannot move out of `foo_const.pointer` which is behind a shared reference
    assert_ty::<T>(x);
}

fn by_ref_non_pinned_unpin<T: Unpin>(foo_mut: &pin mut Foo<T>, foo_const: &pin const Foo<T>) {
    let &pin mut Foo(ref mut x) = foo_mut;
    assert_mut(x);

    let &pin const Foo(ref x) = foo_const;
    assert_ref(x);
}

fn by_ref_non_pinned_non_unpin<T>(foo_mut: &pin mut Foo<T>, foo_const: &pin const Foo<T>) {
    let &pin mut Foo(ref mut x) = foo_mut;
    //~^ ERROR `T` cannot be unpinned
    assert_mut(x);

    let &pin const Foo(ref x) = foo_const;
    //~^ ERROR `T` cannot be unpinned
    assert_ref(x);
}

// Makes sure that `ref pin` binding mode cannot be changed to a `ref` binding mode.
//
// This mimics the following code:
// ```
// fn f<'a, T>(
//     ((&x,),): &'a (&'a mut (&'a T,),),
// ) -> T {
//     x
// }
// ```
fn tuple_tuple_ref_pin_mut_pat_and_pin_mut_of_tuple_mut_of_tuple_pin_mut_ty<'a, T>(
    ((&pin mut x,),): &'a pin mut (&'a mut (&'a pin mut Foo<T>,),),
    //~^ ERROR cannot explicitly dereference within an implicitly-borrowing pattern
    //~| ERROR cannot move out of a mutable reference
) -> Foo<T> {
    x
}

fn tuple_ref_pin_mut_pat_and_mut_of_tuple_pin_mut_ty<'a, T>(
    (&pin mut x,): &'a mut (&'a pin mut Foo<T>,), //~ ERROR `T` cannot be unpinned
) -> &'a mut Foo<T> {
    x
}

fn tuple_ref_pin_mut_pat_and_mut_of_mut_tuple_pin_mut_ty<'a, T>(
    (&pin mut x,): &'a mut &'a pin mut (Foo<T>,), //~ ERROR mismatched type
) -> &'a mut Foo<T> {
    x
}

fn ref_pin_mut_tuple_pat_and_mut_of_tuple_pin_mut_ty<'a, T>(
    &pin mut (x,): &'a mut (&'a pin mut Foo<T>,), //~ ERROR mismatched type
) -> &'a mut Foo<T> {
    x
}

fn ref_pin_mut_tuple_pat_and_mut_of_mut_tuple_pin_mut_ty<'a, T>(
    &pin mut (x,): &'a mut &'a pin mut (Foo<T>,), //~ ERROR mismatched type
) -> &'a mut Foo<T> {
    x
}

fn main() {}
