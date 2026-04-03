// Field lookup still resolves to the public field on the Deref target, but
// follow-up diagnostics should explain that the original type has a same-named
// private field with a different type.
//@ dont-require-annotations: ERROR

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

trait Marker {}

impl Marker for usize {}

struct Wrapper(i32);

impl<T: Marker> std::ops::Add<T> for Wrapper {
    type Output = ();

    fn add(self, _: T) {}
}

fn by_value(a: A) {
    a.field + 5;
}

fn by_ref(a: &A) {
    a.field + 5;
}

fn rhs_by_value(a: A) {
    5 + a.field;
}

fn rhs_by_ref(a: &A) {
    5 + a.field;
}

fn rhs_assign_op_by_value(a: A) {
    let mut n = 5;
    n += a.field;
}

fn rhs_assign_op_by_ref(a: &A) {
    let mut n = 5;
    n += a.field;
}

fn rhs_nested_obligation(a: A) {
    Wrapper(5) + a.field;
}

fn method_call(a: A) {
    a.field.count_ones();
}

fn type_mismatch(a: A) {
    let value: usize = a.field;
    eprintln!("value: {value}");
}

fn function_arg(a: A) {
    takes_usize(a.field);
}

fn return_value(a: &A) -> usize {
    a.field
}

fn main() {}
