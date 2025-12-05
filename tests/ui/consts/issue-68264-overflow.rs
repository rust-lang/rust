//@ check-pass
//@ compile-flags: --emit=mir,link
// Regression test for issue #68264
// Checks that we don't encounter overflow
// when running const-prop on functions with
// complicated bounds
pub trait Query {}

pub trait AsQuery {
    type Query: Query;
}
pub trait Table: AsQuery + Sized {}

pub trait LimitDsl {
    type Output;
}

pub(crate) trait LoadQuery<Conn, U>: RunQueryDsl<Conn> {}

impl<T: Query> AsQuery for T {
    type Query = Self;
}

impl<T> LimitDsl for T
where
    T: Table,
    T::Query: LimitDsl,
{
    type Output = <T::Query as LimitDsl>::Output;
}

pub(crate) trait RunQueryDsl<Conn>: Sized {
    fn first<U>(self, _conn: &Conn) -> U
    where
        Self: LimitDsl,
        Self::Output: LoadQuery<Conn, U>,
    {
        // Overflow is caused by this function body
        unimplemented!()
    }
}

fn main() {}
