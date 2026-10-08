//@ edition: 2024
//@ run-pass

fn foo<'a>(b: &'a ()) -> impl Fn() {
    || {
        expected::<&()>(b);
    }
}

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
