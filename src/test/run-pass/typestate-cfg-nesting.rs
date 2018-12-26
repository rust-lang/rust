#![allow(dead_code)]
#![allow(unused_assignments)]
#![allow(unknown_lints)]
// pretty-expanded FIXME #23616

#![allow(dead_assignment)]
#![allow(unused_variables)]

fn f() {
    let x = 10; let mut y = 11;
    if true { match x { _ => { y = x; } } } else { }
}

pub fn main() {
    let x = 10;
    let mut y = 11;
    if true { while false { y = x; } } else { }
}
