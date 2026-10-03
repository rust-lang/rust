//@ edition: 2024
//@ run-pass

fn foo<'a>(b: &'a ()) -> impl Fn() {
    || {
        expected::<&()>(b);
    }
}

// No reborrow of `b` is emitted which means our closure captures
// `b` by ref resulting in an upvar of `&&'a ()`
fn bar<'a>(b: &'a ()) -> impl Fn() {
    || {
        expected::<&'a ()>(b);
    }
}

fn expected<T>(_: T) {}

fn main() {
    foo(&())();
    bar(&())();
}
