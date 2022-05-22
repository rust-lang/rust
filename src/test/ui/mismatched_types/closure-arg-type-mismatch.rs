// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn main() {
    let a = [(1u32, 2u32)];
    a.iter().map(|_: (u32, u32)| 45); //~ ERROR type mismatch
    a.iter().map(|_: &(u16, u16)| 45); //~ ERROR type mismatch
    a.iter().map(|_: (u16, u16)| 45); //~ ERROR type mismatch
}

fn baz<F: Fn(*mut &u32)>(_: F) {}
fn _test<'a>(f: fn(*mut &'a u32)) {
    baz(f);
    //[base]~^ ERROR implementation of `FnOnce` is not general enough
    //[base]~| ERROR implementation of `FnOnce` is not general enough
    //[base]~| ERROR mismatched types
    //[base]~| ERROR mismatched types
}
