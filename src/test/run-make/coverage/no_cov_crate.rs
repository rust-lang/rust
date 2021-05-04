// Enables `no_coverage` on the entire crate
#![feature(no_coverage)]

#[no_coverage]
fn do_not_add_coverage_1() {
    println!("called but not covered");
}

#[no_coverage]
fn do_not_add_coverage_2() {
    println!("called but not covered");
}

fn main() {
    do_not_add_coverage_1();
    do_not_add_coverage_2();
}
