#![allow(unused)]

fn test(shouldwe: Option<u32>, shouldwe2: Option<u32>) -> u32 {
    //~^ NOTE expected `u32` because of return type
    match shouldwe {
        Some(val) => {
            match shouldwe2 {
                Some(val) => {
                    return val;
                }
                None => (), //~ ERROR mismatched types
                //~^ NOTE expected `u32`, found `()`
            }
        }
        None => return 12,
    }
}

fn main() {
    println!("returned {}", test(None, Some(5)));
}
