//@ run-pass
#![allow(non_upper_case_globals)]

enum A { A1, A2 }
enum B { B1=4, B2=2 }

#[allow(dead_code)]
#[repr(align(8))]
enum Aligned {
    Zero = 0,
    One = 1,
}

// regression test for https://github.com/rust-lang/rust/issues/96185
const X: u8 = {
    let aligned = Aligned::Zero;
    aligned as u8
};

pub fn main () {
    static c1: isize = A::A2 as isize;
    static c2: isize = B::B2 as isize;
    let a1 = A::A2 as isize;
    let a2 = B::B2 as isize;
    assert_eq!(c1, 1);
    assert_eq!(c2, 2);
    assert_eq!(a1, 1);
    assert_eq!(a2, 2);

    // Turns out that adding a let-binding generates totally different MIR.
    static c1_2: isize = { let v = A::A1; v as isize };
    static c2_2: isize = { let v = B::B1; v as isize };
    let a1_2 = { let v = A::A1; v as isize };
    let a2_2 = { let v = B::B1; v as isize };
    assert_eq!(c1_2, 0);
    assert_eq!(c2_2, 4);
    assert_eq!(a1_2, 0);
    assert_eq!(a2_2, 4);

    assert_eq!(X, 0);
}
