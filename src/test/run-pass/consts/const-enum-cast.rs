// run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

enum A { A1, A2 }
enum B { B1=0, B2=2 }

pub fn main () {
    static c1: isize = A::A2 as isize;
    static c2: isize = B::B2 as isize;
    let a1 = A::A2 as isize;
    let a2 = B::B2 as isize;
    assert_eq!(c1, 1);
    assert_eq!(c2, 2);
    assert_eq!(a1, 1);
    assert_eq!(a2, 2);
}
