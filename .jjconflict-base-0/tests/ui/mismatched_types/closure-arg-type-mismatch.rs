fn main() {
    let a = [(1u32, 2u32)];
    a.iter().map(|_: (u32, u32)| 45); //~ ERROR type mismatch
    a.iter().map(|_: &(u16, u16)| 45); //~ ERROR type mismatch
    a.iter().map(|_: (u16, u16)| 45); //~ ERROR type mismatch
}

fn baz<F: Fn(*mut &u32)>(_: F) {}
fn _test<'a>(f: fn(*mut &'a u32)) {
    baz(f);
    //~^ ERROR: implementation of `FnOnce` is not general enough
    //~| ERROR: borrowed data escapes
    //~| ERROR: not general enough
}
