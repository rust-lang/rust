#![allow(warnings)]

fn closure_expecting_bound<F>(_: F)
where
    F: FnOnce(&u32),
{
}

fn expect_bound_supply_named<'x>() {
    let mut f: Option<&u32> = None;

    // Here we give a type annotation that `x` should be free. We get
    // an error because of that.
    closure_expecting_bound(|x: &'x u32| {
        //~^ ERROR lifetime may not live long enough
        //~| ERROR lifetime may not live long enough

        // Borrowck doesn't get a chance to run, but if it did it should error
        // here.
        f = Some(x);
    });
}

fn main() {}
