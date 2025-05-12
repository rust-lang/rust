#![feature(coverage_attribute)]
//@ edition: 2021

#[rustfmt::skip]
fn if_not(cond: bool) {
    if
        !
        cond
    {
        println!("cond was false");
    }

    if
        !
        cond
    {
        println!("cond was false");
    }

    if
        !
        cond
    {
        println!("cond was false");
    } else {
        println!("cond was true");
    }
}

#[coverage(off)]
fn main() {
    for _ in 0..8 {
        if_not(std::hint::black_box(true));
    }
    for _ in 0..4 {
        if_not(std::hint::black_box(false));
    }
}
