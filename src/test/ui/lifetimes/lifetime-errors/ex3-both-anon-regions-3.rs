// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn foo(z: &mut Vec<(&u8,&u8)>, (x, y): (&u8, &u8)) {
    z.push((x,y));
    //[base]~^ ERROR lifetime mismatch
    //[base]~| ERROR lifetime mismatch
    //[nll]~^^^ ERROR lifetime may not live long enough
    //[nll]~| ERROR lifetime may not live long enough
}

fn main() { }
