fn foo<T, U>(x: T, y: U) {
    let mut xx = x;
    xx = y;
    //~^  ERROR mismatched types
    //~| NOTE_NONVIRAL expected type parameter `T`, found type parameter `U`
    //~| NOTE_NONVIRAL expected type parameter `T`
    //~| NOTE_NONVIRAL found type parameter `U`
}

fn main() {
}
