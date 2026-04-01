use std::vec::Vec;

fn main() {
    let a: Vec<isize> = Vec::new();
    a.iter().all(|_| -> bool {
        //~^ ERROR mismatched types
    });
}
