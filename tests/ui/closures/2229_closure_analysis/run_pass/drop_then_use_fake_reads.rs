// edition:2021
// check-pass
#![feature(rustc_attrs)]

fn main() {
    let mut x = 1;
    let c = || {
        drop(&mut x);
        match x { _ => () }
    };
}
