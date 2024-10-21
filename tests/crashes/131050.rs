//@ known-bug: #131050
//@ compile-flags: --edition=2021

fn query_as<D>() {}

async fn create_user() {
    query_as();
}

async fn post_user_filter() -> impl Filter {
    AndThen(&(), || async { create_user().await })
}

async fn get_app() -> impl Send {
    post_user_filter().await
}

trait Filter {}

struct AndThen<T, F>(T, F);

impl<T, F, R> Filter for AndThen<T, F>
where
    F: Fn() -> R,
    R: Send,
{
}
