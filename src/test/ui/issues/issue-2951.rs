fn foo<T, U>(x: T, y: U) {
    let mut xx = x;
    xx = y;
    //~^  ERROR mismatched types
    //~| expected type `T`
    //~| found type `U`
    //~| expected type parameter, found a different type parameter
}

fn main() {
}
