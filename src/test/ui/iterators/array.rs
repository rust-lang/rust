fn main() {
    for _ in [1, 2] {}
//~^ ERROR is not an iterator
    let x = [1, 2];
    for _ in x {}
//~^ ERROR is not an iterator
    for _ in [1.0, 2.0] {}
//~^ ERROR is not an iterator
}
