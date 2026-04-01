//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
fn with_closure<A, F>(_: F)
    where F: FnOnce(Vec<A>, A)
{
}

fn expect_free_supply_free<'x>(x: &'x u32) {
    with_closure(|mut x: Vec<_>, y| {
        // Shows that the call to `x.push()` is influencing type of `y`...
        x.push(22_u32);

        // ...since we now know the type of `y` and can resolve the method call.
        let _ = y.wrapping_add(1);
    });
}

fn main() { }
