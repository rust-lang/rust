fn thing(x: impl FnOnce(&u32, &u32, u32)) {}

fn main() {
    let f = | _ , y: &u32 , z | ();
    thing(f);
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    let f = | x, y: _  , z: u32 | ();
    thing(f);
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `FnOnce` is not general enough
}
