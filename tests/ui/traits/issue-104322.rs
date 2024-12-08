//@ build-pass
//
// Tests that overflows do not occur in certain situations
// related to generic diesel code

use mini_diesel::*;

pub trait HandleDelete<K> {}

pub fn handle_delete<D, R>()
where
    R: HasTable,
    R::Table: HandleDelete<D> + 'static,
{
}

impl<K, T> HandleDelete<K> for T
where
    T: Table + HasTable<Table = T> + 'static,
    K: 'static,
    &'static K: Identifiable<Table = T>,
    T::PrimaryKey: EqAll<<&'static K as Identifiable>::Id>,
    T::Query: FilterDsl<<T::PrimaryKey as EqAll<<&'static K as Identifiable>::Id>>::Output>,
    Filter<T::Query, <T::PrimaryKey as EqAll<<&'static K as Identifiable>::Id>>::Output>:
        IntoUpdateTarget<Table = T>,
{
}

mod mini_diesel {
    pub trait HasTable {
        type Table: Table;
    }

    pub trait Identifiable: HasTable {
        type Id;
    }

    pub trait EqAll<Rhs> {
        type Output;
    }

    pub trait IntoUpdateTarget: HasTable {
        type WhereClause;
    }

    pub trait Query {
        type SqlType;
    }

    pub trait AsQuery {
        type Query: Query;
    }
    impl<T: Query> AsQuery for T {
        type Query = Self;
    }

    pub trait FilterDsl<Predicate> {
        type Output;
    }

    impl<T, Predicate> FilterDsl<Predicate> for T
    where
        T: Table,
        T::Query: FilterDsl<Predicate>,
    {
        type Output = Filter<T::Query, Predicate>;
    }

    pub trait QuerySource {
        type FromClause;
    }

    pub trait Table: QuerySource + AsQuery + Sized {
        type PrimaryKey;
    }

    pub type Filter<Source, Predicate> = <Source as FilterDsl<Predicate>>::Output;
}

fn main() {}
