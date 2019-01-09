#![allow(non_camel_case_types)]
#[derive(Copy, Clone)]
struct mytype(Mytype);

#[derive(Copy, Clone)]
struct Mytype {
    compute: fn(mytype) -> isize,
    val: isize,
}

fn compute(i: mytype) -> isize {
    let mytype(m) = i;
    return m.val + 20;
}

pub fn main() {
    let myval = mytype(Mytype{compute: compute, val: 30});
    println!("{}", compute(myval));
    let mytype(m) = myval;
    assert_eq!((m.compute)(myval), 50);
}
