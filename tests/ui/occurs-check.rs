fn main() {
    let f;
    f = Box::new(f);
    //~^ ERROR overflow setting `Box<_>` to a subtype of `_`
}
