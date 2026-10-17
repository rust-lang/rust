// Regression test for #80953. Hitting the recursion limit in projection
// is non-fatal. The above code, minimised from wundergraph shows a case
// where this is relied on.

//@ check-pass

struct AlternateTable {}
struct AlternateQuery {}

pub trait Query {}
pub trait AsQuery {
    type Query;
}
impl<T: Query> AsQuery for T {
    type Query = Self;
}
impl AsQuery for AlternateTable {
    type Query = AlternateQuery;
}

pub trait Table: AsQuery {
    type PrimaryKey;
}
impl Table for AlternateTable {
    type PrimaryKey = ();
}

pub trait FilterDsl<Predicate> {
    type Output;
}
pub type Filter<Source, Predicate> = <Source as FilterDsl<Predicate>>::Output;
impl<T, Predicate> FilterDsl<Predicate> for T
where
    T: Table,
    T::Query: FilterDsl<Predicate>,
{
    type Output = Filter<T::Query, Predicate>;
}
impl<Predicate> FilterDsl<Predicate> for AlternateQuery {
    type Output = &'static str;
}

pub trait HandleDelete {
    type Filter;
}
impl<T> HandleDelete for T
where
    T: Table,
    T::Query: FilterDsl<T::PrimaryKey>,
    Filter<T::Query, T::PrimaryKey>: ,
{
    type Filter = Filter<T::Query, T::PrimaryKey>;
}

fn main() {
    let x: <AlternateTable as HandleDelete>::Filter = "Hello, world";
    println!("{}", x);
}
