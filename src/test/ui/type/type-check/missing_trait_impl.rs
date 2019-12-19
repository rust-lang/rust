fn main() {
}

fn foo<T>(x: T, y: T) {
    let z = x + y; //~ ERROR cannot add `T` to `T`
}

fn bar<T>(x: T) {
    x += x; //~ ERROR binary assignment operation `+=` cannot be applied to type `T`
}
