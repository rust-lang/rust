#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]

fn main() {
    mut_range_bound_upper();
    mut_range_bound_lower();
    mut_range_bound_both();
    mut_range_bound_no_mutation();
    immut_range_bound();
}

fn mut_range_bound_upper() {
    let mut m = 4;
    for i in 0..m { m = 5; } // warning    
}

fn mut_range_bound_lower() {
    let mut m = 4;
    for i in m..10 { m *= 2; } // warning
}

fn mut_range_bound_both() {
    let mut m = 4;
    let mut n = 6;
    for i in m..n { m = 5; n = 7; } // warning (1 for each mutated bound)
}

fn mut_range_bound_no_mutation() {
    let mut m = 4;
    for i in 0..m { continue; } // no warning
}

fn mut_borrow_range_bound() {
    let mut m = 4;
    for i in 0..m {
        let n = &mut m;
        *n += 1;
    }
}


fn immut_range_bound() {
    let m = 4;
    for i in 0..m { continue; } // no warning
}
