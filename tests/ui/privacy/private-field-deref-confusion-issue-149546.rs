// Field lookup still resolves to the public field on the Deref target, but
// follow-up diagnostics should explain that the original type has a same-named
// private field with a different type.

mod structs {
    pub struct A {
        field: usize,
        b: B,
    }

    pub struct B {
        pub field: bool,
    }

    impl std::ops::Deref for A {
        type Target = B;

        fn deref(&self) -> &Self::Target {
            &self.b
        }
    }
}

use structs::A;

fn takes_usize(_: usize) {}
//~^ NOTE function defined here

trait Marker {}

impl Marker for usize {}
//~^ HELP the trait `Marker` is implemented for `usize`

struct Wrapper(i32);

impl<T: Marker> std::ops::Add<T> for Wrapper {
    //~^ NOTE required for `Wrapper` to implement `Add<bool>`
    //~| NOTE unsatisfied trait bound introduced here
    type Output = ();

    fn add(self, _: T) {}
}

fn by_value(a: A) {
    a.field + 5;
    //~^ ERROR cannot add `{integer}` to `bool`
    //~| NOTE bool
    //~| NOTE {integer}
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn by_ref(a: &A) {
    a.field + 5;
    //~^ ERROR cannot add `{integer}` to `bool`
    //~| NOTE bool
    //~| NOTE {integer}
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn rhs_by_value(a: A) {
    5 + a.field;
    //~^ ERROR cannot add `bool` to `{integer}`
    //~| NOTE no implementation for `{integer} + bool`
    //~| HELP the trait `Add<bool>` is not implemented for `{integer}`
    //~| HELP the following other types implement trait `Add<Rhs>`:
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn rhs_by_ref(a: &A) {
    5 + a.field;
    //~^ ERROR cannot add `bool` to `{integer}`
    //~| NOTE no implementation for `{integer} + bool`
    //~| HELP the trait `Add<bool>` is not implemented for `{integer}`
    //~| HELP the following other types implement trait `Add<Rhs>`:
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn rhs_assign_op_by_value(a: A) {
    let mut n = 5;
    n += a.field;
    //~^ ERROR cannot add-assign `bool` to `{integer}`
    //~| NOTE no implementation for `{integer} += bool`
    //~| HELP the trait `AddAssign<bool>` is not implemented for `{integer}`
    //~| HELP the following other types implement trait `AddAssign<Rhs>`:
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn rhs_assign_op_by_ref(a: &A) {
    let mut n = 5;
    n += a.field;
    //~^ ERROR cannot add-assign `bool` to `{integer}`
    //~| NOTE no implementation for `{integer} += bool`
    //~| HELP the trait `AddAssign<bool>` is not implemented for `{integer}`
    //~| HELP the following other types implement trait `AddAssign<Rhs>`:
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn rhs_nested_obligation(a: A) {
    Wrapper(5) + a.field;
    //~^ ERROR the trait bound `bool: Marker` is not satisfied
    //~| NOTE the trait `Marker` is not implemented for `bool`
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn method_call(a: A) {
    a.field.count_ones();
    //~^ ERROR no method named `count_ones` found for type `bool` in the current scope
    //~| NOTE method not found in `bool`
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn type_mismatch(a: A) {
    let value: usize = a.field;
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `bool`
    //~| NOTE expected due to this
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
    eprintln!("value: {value}");
}

fn function_arg(a: A) {
    takes_usize(a.field);
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `bool`
    //~| NOTE arguments to this function are incorrect
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn return_value(a: &A) -> usize {
    //~^ NOTE expected `usize` because of return type
    a.field
    //~^ ERROR mismatched types
    //~| NOTE expected `usize`, found `bool`
    //~| NOTE there is a field `field` on `A` with type `usize`, but it is private
}

fn main() {}
