#![feature(box_patterns)]

//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:or-pattern-paren.pp

macro_rules! or_pat {
    ($($name:pat),+) => { $($name)|+ }
}

fn check_at(x: Option<i32>) {
    match x {
        Some(v @ or_pat!(1, 2, 3)) => println!("{v}"),
        _ => {}
    }
}

fn check_ref(x: &i32) {
    match x {
        &or_pat!(1, 2, 3) => {}
        _ => {}
    }
}

fn check_box(x: Box<i32>) {
    match x {
        box or_pat!(1, 2, 3) => {}
        _ => {}
    }
}

fn main() {
    check_at(Some(2));
    check_ref(&1);
    check_box(Box::new(1));
}
