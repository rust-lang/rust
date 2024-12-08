fn thing(x: impl FnOnce(&u32)) {}

fn main() {
    let f = |_| ();
    thing(f);
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
}
