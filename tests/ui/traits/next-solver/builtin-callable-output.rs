//@ run-pass
//@ compile-flags: -Znext-solver=globally

fn once<A, R, F: FnOnce(A) -> R>(f: F, arg: A) -> R {
    f(arg)
}

fn mutable<A, R, F: FnMut(A) -> R>(mut f: F, arg: A) -> R {
    f(arg)
}

fn shared<A, R, F: Fn(A) -> R>(f: F, arg: A) -> R {
    f(arg)
}

fn without_arguments<R, F: FnOnce() -> R>(f: F) -> R {
    f()
}

fn two_arguments<A, B, R, F: FnOnce(A, B) -> R>(f: F, left: A, right: B) -> R {
    f(left, right)
}

fn identity<T>(value: T) -> T {
    value
}

fn borrow<'a>(value: &'a u32) -> &'a u32 {
    value
}

fn first<'a, 'b>(left: &'a u32, _: &'b u32) -> &'a u32 {
    left
}

fn default_value<T: Default>() -> T {
    T::default()
}

trait Family {
    type Item;
}

impl Family for u32 {
    type Item = u32;
}

fn projected<T: Family>(value: T::Item) -> T::Item {
    value
}

fn main() {
    let value = 17;
    assert_eq!(*once(borrow, &value), 17);
    assert_eq!(*mutable(borrow, &value), 17);
    assert_eq!(*shared(borrow, &value), 17);

    let pointer: for<'a> fn(&'a u32) -> &'a u32 = borrow;
    assert_eq!(*once(pointer, &value), 17);
    assert_eq!(*mutable(pointer, &value), 17);
    assert_eq!(*shared(pointer, &value), 17);

    let other = 23;
    let pointer: for<'a, 'b> fn(&'a u32, &'b u32) -> &'a u32 = first;
    assert_eq!(*two_arguments(first, &value, &other), 17);
    assert_eq!(*two_arguments(pointer, &value, &other), 17);

    // The result can constrain a function item's still-unknown generic argument.
    let make = default_value;
    let result: u16 = without_arguments(make);
    assert_eq!(result, 0);
    let pass = identity;
    let result: u64 = once(pass, 31);
    assert_eq!(result, 31);

    // Normalizing the signature may itself require an associated-type goal.
    assert_eq!(once(projected::<u32>, 37), 37);
}
