//@ dont-require-annotations: NOTE

fn foo<T, U>(x: T, y: U) {
    let mut xx = x;
    xx = y;
    //~^  ERROR mismatched types
    //~| NOTE expected type parameter `T`, found type parameter `U`
    //~| NOTE expected type parameter `T`
    //~| NOTE found type parameter `U`
}

fn main() {
}
