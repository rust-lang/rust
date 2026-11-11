// Regression test for #97405.
// In `good_generic_fn` the param `T` ends up in the substs of closures/coroutines,
// but we should be able to prove `<Gen<T> as Iterator>::Item: 'static` without
// requiring `T: 'static`

//@ edition:2018
//@ check-fail

fn opaque<F>(_: F) -> impl Iterator { b"".iter() }

fn assert_static<T: 'static>(_: T) {}

fn good_generic_fn<T>() {
    // Previously, proving `<OpaqueTy<type_of(async {})> as Iterator>::Item: 'static`
    // used to require `T: 'static`.
    assert_static(opaque(async {}).next());
    assert_static(opaque(|| {}).next());
    assert_static(opaque(opaque(async {}).next()).next());
}


// This should fail because `T` ends up in the upvars of the closure.
fn bad_generic_fn<T: Copy>(t: T) {
    assert_static(opaque(async move { t; }).next());
    //~^ ERROR the associated type `<impl Iterator as Iterator>::Item` may not live long enough
    assert_static(opaque(move || { t; }).next());
    //~^ ERROR the associated type `<impl Iterator as Iterator>::Item` may not live long enough
    assert_static(opaque(opaque(async move { t; }).next()).next());
    //~^ ERROR the associated type `<impl Iterator as Iterator>::Item` may not live long enough
}

fn main() {}
