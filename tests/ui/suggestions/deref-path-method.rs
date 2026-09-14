fn main() {
    let vec = Vec::new();
    Vec::contains(&vec, &0);
    //~^ ERROR no associated function or constant named `contains` found for struct `Vec<_, _>` in the current scope
    //~| HELP the function `contains` is implemented on `[_]`
}
