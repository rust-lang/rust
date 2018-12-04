fn main() {
    for _ in [0..1] {}
//~^ ERROR is not an iterator
    for _ in [0..=1] {}
//~^ ERROR is not an iterator
    for _ in [0..] {}
//~^ ERROR is not an iterator
    for _ in [..1] {}
//~^ ERROR is not an iterator
    for _ in [..=1] {}
//~^ ERROR is not an iterator
    let start = 0;
    let end = 0;
    for _ in [start..end] {}
//~^ ERROR is not an iterator
    let array_of_range = [start..end];
    for _ in array_of_range {}
//~^ ERROR is not an iterator
    for _ in [0..1, 2..3] {}
//~^ ERROR is not an iterator
    for _ in [0..=1] {}
//~^ ERROR is not an iterator
}
