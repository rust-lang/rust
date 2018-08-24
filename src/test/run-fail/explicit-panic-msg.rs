#![allow(unused_assignments)]
#![allow(unused_variables)]

// error-pattern:wooooo
fn main() {
    let mut a = 1;
    if 1 == 1 {
        a = 2;
    }
    panic!(format!("woooo{}", "o"));
}
