// Regression test for issue #149546.
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

fn main() {}
