fn main() {
    let f;
    f = Box::new(f);
    //~^ ERROR overflow evaluating the requirement `Box<_> <: _`
}
