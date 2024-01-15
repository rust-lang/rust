fn main() {

    let f;
    let g;

    g = f;
    f = Box::new(g);
    //~^ ERROR  overflow setting `Box<_>` to a subtype of `_`
}
