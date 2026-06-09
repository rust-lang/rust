fn with_closure_expecting_fn_with_free_region<F>(_: F)
where
    F: for<'a> FnOnce(fn(&'a u32), &i32),
{
}

fn with_closure_expecting_fn_with_bound_region<F>(_: F)
where
    F: FnOnce(fn(&u32), &i32),
{
}

fn expect_free_supply_free_from_fn<'x>(x: &'x u32) {
    // Here, the type given for `'x` "obscures" a region from the
    // expected signature that is bound at closure level.
    with_closure_expecting_fn_with_free_region(|x: fn(&'x u32), y| {});
    //~^ ERROR lifetime may not live long enough
    //~| ERROR lifetime may not live long enough
}

fn expect_free_supply_free_from_closure() {
    // A variant on the previous test. Here, the region `'a` will be
    // bound at the closure level, just as is expected, so no error
    // results.
    type Foo<'a> = fn(&'a u32);
    with_closure_expecting_fn_with_free_region(|_x: Foo<'_>, y| {});
}

fn expect_free_supply_bound() {
    // Here, we are given a function whose region is bound at closure level,
    // but we expect one bound in the argument. Error results.
    with_closure_expecting_fn_with_free_region(|x: fn(&u32), y| {});
    //~^ ERROR mismatched types [E0308]
    //~| ERROR lifetime may not live long enough
}

fn expect_bound_supply_free_from_fn<'x>(x: &'x u32) {
    // Here, we are given a `fn(&u32)` but we expect a `fn(&'x
    // u32)`. In principle, this could be ok, but we demand equality.
    with_closure_expecting_fn_with_bound_region(|x: fn(&'x u32), y| {});
    //~^ ERROR mismatched types [E0308]
    //~| ERROR lifetime may not live long enough
}

fn expect_bound_supply_free_from_closure() {
    // A variant on the previous test. Here, the region `'a` will be
    // bound at the closure level, but we expect something bound at
    // the argument level.
    type Foo<'a> = fn(&'a u32);
    with_closure_expecting_fn_with_bound_region(|x: Foo<'_>, y| {
        //~^ ERROR mismatched types
    });
}

fn expect_bound_supply_bound<'x>(x: &'x u32) {
    // No error in this case. The supplied type supplies the bound
    // regions, and hence we are able to figure out the type of `y`
    // from the expected type
    with_closure_expecting_fn_with_bound_region(|x: for<'z> fn(&'z u32), y| {});
}

fn main() {}
