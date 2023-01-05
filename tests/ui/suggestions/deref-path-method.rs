fn main() {
    let vec = Vec::new();
    Vec::contains(&vec, &0);
    //~^ ERROR no function or associated item named `contains` found for struct `Vec<_, _>` in the current scope
    //~| HELP the function `contains` is implemented on `[_]`
}
