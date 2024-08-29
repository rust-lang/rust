//@ known-bug: #127737
//@ compile-flags: -Zmir-opt-level=5 --crate-type lib

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
