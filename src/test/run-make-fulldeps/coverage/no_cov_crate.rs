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

#[no_coverage]
fn do_not_add_coverage_not_called() {
    println!("not called and not covered");
}

fn add_coverage_1() {
    println!("called and covered");
}

fn add_coverage_2() {
    println!("called and covered");
}

fn add_coverage_not_called() {
    println!("not called but covered");
}

fn main() {
    do_not_add_coverage_1();
    do_not_add_coverage_2();
    add_coverage_1();
    add_coverage_2();
}
