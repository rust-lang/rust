fn main() {

    let f;
    let g;

    g = f;
    f = Box::new(g);
    //~^  ERROR mismatched types
    //~| cyclic type of infinite size
}
