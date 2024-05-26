//@ known-bug: rust-lang/rust#123276
//@ edition:2021

async fn create_task() {
    _ = Some(async { bind(documentation_filter()) });
}

async fn bind<Fut, F: Filter<Future = Fut>>(_: F) {}

fn documentation_filter() -> impl Filter {
    AndThen
}

trait Filter {
    type Future;
}

struct AndThen;

impl Filter for AndThen
where
    Foo: Filter,
{
    type Future = ();
}
