// build-pass

#![feature(dyn_star)]
//~^ WARN the feature `dyn_star` is incomplete and may not be safe to use and/or cause compiler crashes

fn main() {
    let x: dyn* Send = &();
    let x = Box::new(x) as Box<dyn Send>;
}
