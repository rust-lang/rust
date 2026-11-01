//@ check-fail

#![deny(dropping_copy_types)]

fn main() {
    let y = 1;
    let z = 2;
    match y {
        0 => drop(y), //~ ERROR calls to `std::mem::drop`
        1 => drop(z), //~ ERROR calls to `std::mem::drop`
        2 => drop(3), //~ ERROR calls to `std::mem::drop`
        _ => {},
    }
}
