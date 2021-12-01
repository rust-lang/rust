// check-pass
// Regresion test for <https://github.com/rust-lang/rust/issues/79459>.
pub trait Query {}

pub trait AsQuery {
    type Query;
}

impl<T: Query> AsQuery for T {
    type Query = T;
}

pub trait SelectDsl<Selection> {
    type Output;
}

impl<T, Selection> SelectDsl<Selection> for T
where
    T: AsQuery,
    T::Query: SelectDsl<Selection>,
{
    type Output = <T::Query as SelectDsl<Selection>>::Output;
}

pub type Select<Source, Selection> = <Source as SelectDsl<Selection>>::Output;
