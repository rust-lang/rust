//@ check-pass
// Found in a crater run on #118553

pub trait QueryBase {
    type Db;
}

pub trait AsyncQueryFunction<'f>: // 'f is important
    QueryBase<Db = <Self as AsyncQueryFunction<'f>>::SendDb> // bound is important
{
    type SendDb;
}

pub struct QueryTable<'me, Q, DB> {
    _q: Option<Q>,
    _db: Option<DB>,
    _marker: Option<&'me ()>,
}

impl<'me, Q> QueryTable<'me, Q, <Q as QueryBase>::Db> // projection is important
//   ^^^ removing 'me (and in QueryTable) gives a different error
where
    Q: for<'f> AsyncQueryFunction<'f>,
{
    pub fn get_async<'a>(&'a mut self) {
        panic!();
    }
}

fn main() {}
