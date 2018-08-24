#![feature(slice_patterns)]

fn main() {
    let r = &[1, 2];
    match r {
        &[a, b, c, rest..] => {
        //~^ ERROR E0528
        }
    }
}
