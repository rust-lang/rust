use std::alloc::Allocator;
use std::marker::PhantomData;

use serde_with::SerializeAs;

#[macro_export]
macro_rules! vec_in {
    (in: $alloc:expr $(,)?) => (
        ::std::vec::Vec::new_in($alloc)
    );
    (in: $alloc:expr, $elem:expr; $n:expr) => (
        ::std::vec::from_elem_in($elem, $n, $alloc)
    );
    (in: $alloc:expr, $($x:expr),+ $(,)?) => ({
        let alloc = $alloc;
        let mut v = ::std::vec::Vec::with_capacity_in(
            const { [$($crate::vec_in!(@count $x)),+].len() },
            alloc,
        );
        $(v.push($x);)+
        v
    });
    (@count $_x:expr) => { () };
}

pub(crate) trait FromIteratorWithAlloc<A: Allocator, E>: Sized {
    fn from_iter_with_alloc<T>(iter: T, alloc: A) -> Self
    where
        T: IntoIterator<Item = E>;
}

impl<A: Allocator, E> FromIteratorWithAlloc<A, E> for Vec<E, A> {
    fn from_iter_with_alloc<T>(iter: T, alloc: A) -> Self
    where
        T: IntoIterator<Item = E>,
    {
        let mut vec = Vec::new_in(alloc);
        iter.into_iter().collect_into(&mut vec);
        vec
    }
}

pub(crate) trait CollectIn: Iterator {
    fn collect_in<A: Allocator + Copy, B: FromIteratorWithAlloc<A, Self::Item>>(self, alloc: A) -> B
    where
        Self: Sized,
    {
        FromIteratorWithAlloc::from_iter_with_alloc(self, alloc)
    }
}

impl<T> CollectIn for T where T: Iterator {}

pub struct AsSlice<T>(PhantomData<fn() -> [T]>);

impl<T, U, A> SerializeAs<Vec<T, A>> for AsSlice<U>
where
    U: SerializeAs<T>,
    A: Allocator,
{
    fn serialize_as<S>(source: &Vec<T, A>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        <[U]>::serialize_as(source, serializer)
    }
}
