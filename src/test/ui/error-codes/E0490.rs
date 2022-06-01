// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

fn f<'a, 'b>(y: &'b ()) {
    let x: &'a _ = &y;
    //[base]~^ E0490
    //[base]~| E0495
    //[base]~| E0495
    //[nll]~^^^^ lifetime may not live long enough
    //[nll]~| E0597
}

fn main() {}
