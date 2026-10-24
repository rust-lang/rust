fn main() {
    let vec = Vec::new();
    Vec::contains(&vec, &0);
    //~^ ERROR `Vec<_, _>` is not an iterator
    //~| HELP the function `contains` is implemented on `[_]`
}
