fn foo<T, U>(x: T, y: U) {
    let mut xx = x;
    xx = y;
    //~^  ERROR mismatched types
    //~| expected type parameter `T`, found type parameter `U`
    //~| expected type parameter `T`
    //~| found type parameter `U`
}

fn main() {
}
