fn same_type<T>(_: T, _: T) {}

fn f<X, Y>(x: X, y: Y) {
    same_type([x], Some(y));
    //~^ ERROR mismatched types
}

fn main() {}
