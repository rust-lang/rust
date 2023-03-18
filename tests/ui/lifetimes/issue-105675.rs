fn thing(x: impl FnOnce(&u32, &u32)) {}

fn main() {
    let f = |_, _| ();
    thing(f);
    //~^ ERROR mismatched types
    //~^^ ERROR mismatched types
    //~^^^ ERROR implementation of `FnOnce` is not general enough
    //~^^^^ ERROR implementation of `FnOnce` is not general enough
    let f = |x, y| ();
    thing(f);
    //~^ ERROR mismatched types
    //~^^ ERROR mismatched types
    //~^^^ ERROR implementation of `FnOnce` is not general enough
    //~^^^^ ERROR implementation of `FnOnce` is not general enough
}
