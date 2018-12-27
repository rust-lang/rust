#![feature(crate_in_paths)]

fn main() {
    let crate = 0;
    //~^ ERROR expected unit struct/variant or constant, found module `crate`
}
