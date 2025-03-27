fn main() {

    let f;
    let g;

    g = f;
    //~^ ERROR overflow assigning `Box<_>` to `_`
    f = Box::new(g);
}
