fn main() {
    let f;
    f = Box::new(f);
    //~^ ERROR overflow assigning `Box<_>` to `_`
}
