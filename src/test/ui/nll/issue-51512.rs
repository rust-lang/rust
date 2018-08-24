#![allow(warnings)]
#![feature(nll)]

fn main() {
    let range = 0..1;
    let r = range;
    let x = range.start;
    //~^ ERROR use of moved value: `range.start` [E0382]
}
