//@ check-pass

// Regression test for #112832.
pub trait QueryDb {
    type Db;
}

pub struct QueryTable<Q, DB> {
    db: DB,
    storage: Q,
}

// We normalize `<Q as QueryDb>::Db` to `<Q as AsyncQueryFunction<'d>>::SendDb`
// using the where-bound. 'd is an unconstrained region variable which previously
// triggered an assert.
impl<Q> QueryTable<Q, <Q as QueryDb>::Db> where Q: for<'d> AsyncQueryFunction<'d> {}

pub trait AsyncQueryFunction<'d>: QueryDb<Db = <Self as AsyncQueryFunction<'d>>::SendDb> {
    type SendDb: 'd;
}

pub trait QueryStorageOpsAsync<Q>
where
    Q: for<'d> AsyncQueryFunction<'d>,
{
}

fn main() {}
