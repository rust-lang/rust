pub trait Backend {}
pub trait SupportsDefaultKeyword {}

impl SupportsDefaultKeyword for Postgres {}

pub struct Postgres;

impl Backend for Postgres {}

pub struct AstPass<DB>(::std::marker::PhantomData<DB>);

pub trait QueryFragment<DB: Backend> {}


#[derive(Debug, Clone, Copy)]
pub struct BatchInsert<'a, T: 'a, Tab> {
    _marker: ::std::marker::PhantomData<(&'a T, Tab)>,
}

impl<'a, T:'a, Tab, DB> QueryFragment<DB> for BatchInsert<'a, T, Tab>
where DB: SupportsDefaultKeyword + Backend,
{}

pub trait LibToOwned {
    type Owned;
}

pub struct LibCow<T: LibToOwned, Owned = <T as LibToOwned>::Owned> {
    pub t: T,
    pub o: Owned,
}
