// check-pass

use std::marker::PhantomData;

pub trait Scalar: 'static {
    type RefType<'a>: ScalarRef<'a>;
}

pub trait ScalarRef<'a>: 'a {}

impl Scalar for i32 {
    type RefType<'a> = i32;
}

impl Scalar for String {
    type RefType<'a> = &'a str;
}

impl Scalar for bool {
    type RefType<'a> = i32;
}

impl<'a> ScalarRef<'a> for bool {}

impl<'a> ScalarRef<'a> for i32 {}

impl<'a> ScalarRef<'a> for &'a str {}

fn str_contains(a: &str, b: &str) -> bool {
    a.contains(b)
}

pub struct BinaryExpression<A: Scalar, B: Scalar, O: Scalar, F>
where
    F: Fn(A::RefType<'_>, B::RefType<'_>) -> O,
{
    f: F,
    _phantom: PhantomData<(A, B, O)>,
}

impl<A: Scalar, B: Scalar, O: Scalar, F> BinaryExpression<A, B, O, F>
where
    F: Fn(A::RefType<'_>, B::RefType<'_>) -> O,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            _phantom: PhantomData,
        }
    }
}

fn main() {
    BinaryExpression::<String, String, bool, _>::new(str_contains);
}
