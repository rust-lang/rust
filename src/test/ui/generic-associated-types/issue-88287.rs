// check-pass
// edition:2018

#![feature(generic_associated_types)]
#![feature(type_alias_impl_trait)]

use std::future::Future;

trait SearchableResource<Criteria> {
    type SearchResult;
}

trait SearchableResourceExt<Criteria>: SearchableResource<Criteria> {
    type Future<'f, A: 'f + ?Sized, B: 'f>: Future<Output = Result<Vec<A::SearchResult>, ()>> + 'f
    where
        A: SearchableResource<B>,
        Self: 'f;

    fn search<'c>(&'c self, client: &'c ()) -> Self::Future<'c, Self, Criteria>;
}

type SearchFutureTy<'f, A, B: 'f>
where
    A: SearchableResource<B> + ?Sized + 'f,
= impl Future<Output = Result<Vec<A::SearchResult>, ()>> + 'f;
impl<T, Criteria> SearchableResourceExt<Criteria> for T
where
    T: SearchableResource<Criteria>,
{
    type Future<'f, A, B: 'f>
    where
        A: SearchableResource<B> + ?Sized + 'f,
        Self: 'f,
    = SearchFutureTy<'f, A, B>;

    fn search<'c>(&'c self, _client: &'c ()) -> Self::Future<'c, Self, Criteria> {
        async move { todo!() }
    }
}

fn main() {}
