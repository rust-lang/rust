//@ compile-flags: --crate-type=lib -Zmir-opt-level=2
//@ build-pass
// ^-- Must be build-pass, because check-pass will not run const prop.

pub trait TestTrait {
    type MyType;
    fn func() -> Option<Self>
    where
        Self: Sized;
}

impl<T> dyn TestTrait<MyType = T>
where
    Self: Sized,
{
    pub fn other_func() -> Option<Self> {
        match Self::func() {
            Some(me) => Some(me),
            None => None,
        }
    }
}
