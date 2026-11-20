//@ revisions: stock hr
//@[hr] compile-flags: -Zhigher-ranked-assumptions
//@ edition: 2024
//@ check-pass

// Test that we don't normalize the higher-ranked assumptions of an auto trait goal
// unless we have `-Zhigher-ranked-assumptions`, since obligations that result from
// this normalization may lead to higher-ranked lifetime errors when the flag is not
// enabled.

// Regression test for <https://github.com/rust-lang/rust/issues/147244>.

pub fn a() -> impl Future + Send {
    async {
        let queries = core::iter::empty().map(Thing::f);
        b(queries).await;
    }
}

async fn b(queries: impl IntoIterator) {
    c(queries).await;
}

fn c<'a, I>(_queries: I) -> impl Future
where
    I: IntoIterator,
    I::IntoIter: 'a,
{
    async {}
}

pub struct Thing<'a>(pub &'a ());

impl Thing<'_> {
    fn f(_: &Self) {}
}

fn main() {}
