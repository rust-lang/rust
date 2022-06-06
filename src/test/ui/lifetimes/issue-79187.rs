// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn thing(x: impl FnOnce(&u32)) {}

fn main() {
    let f = |_| ();
    thing(f);
    //[nll]~^ ERROR mismatched types
    //~^^ ERROR implementation of `FnOnce` is not general enough
}
