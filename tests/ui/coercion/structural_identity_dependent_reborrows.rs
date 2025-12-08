//@ edition: 2024

// We avoid emitting reborrow coercions if it seems like it would
// not result in a different lifetime on the borrow. This can effect
// capture analysis resulting in borrow checking errors.

fn foo<'a>(b: &'a ()) -> impl Fn() {
    || {
        expected::<&()>(b);
    }
}

// No reborrow of `b` is emitted which means our closure captures
// `b` by ref resulting in an upvar of `&&'a ()`
fn bar<'a>(b: &'a ()) -> impl Fn() {
    || {
        //~^ ERROR: closure may outlive the current function
        expected::<&'a ()>(b);
    }
}

fn expected<T>(_: T) {}

fn main() {}
