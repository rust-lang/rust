fn main() {

    let f;
    let g;

    g = f;
    f = Box::new(g);
    //~^ ERROR overflow evaluating the requirement `Box<_> <: _`
}
