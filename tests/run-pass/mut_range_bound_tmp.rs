#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]

fn main() {
    mut_range_bound_upper();
    mut_range_bound_lower();
    mut_range_bound_both();
    immut_range_bound();
}

fn mut_range_bound_upper() {
    let mut m = 4;
    for i in 0..m { 

        m = 5;
        continue; } // WARNING the range upper bound is mutable
}

fn mut_range_bound_lower() {
    let mut m = 4;
    for i in m..10 { continue; } // WARNING the range lower bound is mutable
}

fn mut_range_bound_both() {
    let mut m = 4;
    let mut n = 6;
    for i in m..n { continue; } // WARNING both bounds are mutable (should get just one warning for this)
}

fn immut_range_bound() {
    let m = 4;
    for i in 0..m { continue; } // no warning
}
