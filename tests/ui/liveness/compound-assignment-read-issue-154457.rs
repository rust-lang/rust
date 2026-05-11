#![deny(unused_assignments)]
#![deny(unused_variables)]
#![allow(dead_code)]

fn compound_assignment_chain() {
    let mut a = 10.;
    //~^ ERROR variable `a` is assigned to, but never used
    let b = 13.;
    let c = 11.;

    a += b;
    a -= c;
    //~^ ERROR value assigned to `a` is never read
}

fn assignment_then_compound_assignment() {
    let mut a;
    //~^ ERROR variable `a` is assigned to, but never used

    a = 10;
    a += 1;
    //~^ ERROR value assigned to `a` is never read
}

fn compound_assignment_chain_with_later_use() {
    let mut a = 10;
    let b = 13;
    let c = 11;

    a += b;
    a -= c;

    let _ = a;
}

fn compound_assignment_after_branch_join(cond: bool) {
    let mut a = 0.0;
    let _ = a;

    if cond {
        a = 1.0;
    } else {
        a = 2.0;
    }

    a += 1.0;
    //~^ ERROR value assigned to `a` is never read
}

fn main() {}
