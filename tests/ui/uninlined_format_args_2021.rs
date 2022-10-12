// run-rustfix
// edition:2021

#![warn(clippy::uninlined_format_args)]

fn main() {
    let var = 1;

    println!("val='{}'", var);

    if var > 0 {
        panic!("p1 {}", var);
    }
    if var > 0 {
        panic!("p2 {0}", var);
    }
    if var > 0 {
        panic!("p3 {var}", var = var);
    }
    if var > 0 {
        panic!("p4 {var}");
    }
}
