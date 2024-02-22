fn main() {

    let f;
    let g;

    g = f;
    f = Box::new(g);
    //~^ ERROR overflow assigning `Box<_>` to `_`
}
