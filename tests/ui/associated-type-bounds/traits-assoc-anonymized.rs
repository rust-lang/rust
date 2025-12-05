//@ check-pass

pub struct LookupInternedStorage;

impl<Q> QueryStorageOps<Q> for LookupInternedStorage
where
    Q: Query,
    for<'d> Q: QueryDb<'d>,
{
    fn fmt_index(&self, db: &<Q as QueryDb<'_>>::DynDb) {
        <<Q as QueryDb<'_>>::DynDb as HasQueryGroup<Q::Group>>::group_storage(db);
    }
}

pub trait HasQueryGroup<G> {
    fn group_storage(&self);
}

pub trait QueryStorageOps<Q>
where
    Q: Query,
{
    fn fmt_index(&self, db: &<Q as QueryDb<'_>>::DynDb);
}

pub trait QueryDb<'d> {
    type DynDb: HasQueryGroup<Self::Group> + 'd;
    type Group;
}

pub trait Query: for<'d> QueryDb<'d> {}

fn main() {}
