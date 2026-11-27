//@ check-pass
//@ edition: 2021
// issue: 113796


trait AsyncLendingIterator {
    type Item<'a>
    where
        Self: 'a;

    #[allow(async_fn_in_trait)]
    async fn next(&mut self) -> Option<Self::Item<'_>>;
}

struct Lend<I>(I);
impl<I> AsyncLendingIterator for Lend<I> {
    type Item<'a> = &'a I
    where
        Self: 'a;

    // Checking that the synthetic `<Self as AsyncLendingIterator>::next()` GAT
    // is well-formed requires being able to assume the WF types of `next`.

    async fn next(&mut self) -> Option<Self::Item<'_>> {
        todo!()
    }
}

fn main() {}
