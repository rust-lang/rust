// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn foo(&mut (ref mut v, w): &mut (&u8, &u8), x: &u8) {
    *v = x;
    //[base]~^ ERROR lifetime mismatch
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() { }
