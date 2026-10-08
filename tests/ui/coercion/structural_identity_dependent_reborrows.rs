//@ edition: 2024

// We avoid emitting reborrow coercions if it seems like it would
// not result in a different lifetime on the borrow. This can effect
// capture analysis resulting in borrow checking errors.

fn foo<'a>(b: &'a mut ()) -> impl FnMut() {
    || {
        expected::<&mut ()>(b);
    }
}

// No reborrow of `b` is emitted which means our closure captures
// `b` by ref resulting in an upvar of `&mut &'a mut ()`
fn bar<'a>(b: &'a mut ()) -> impl FnMut() {
    || {
        expected::<&'a mut ()>(b);
        //~^ ERROR: lifetime may not live long enough
    }
}

fn expected<T>(_: T) {}

fn main() {}
