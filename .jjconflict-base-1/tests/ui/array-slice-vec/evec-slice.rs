//@ run-pass
#![allow(unused_assignments)]

pub fn main() {
    let x : &[isize] = &[1,2,3,4,5];
    let mut z : &[isize] = &[1,2,3,4,5];
    z = x;
    assert_eq!(z[0], 1);
    assert_eq!(z[4], 5);

    let a : &[isize] = &[1,1,1,1,1];
    let b : &[isize] = &[2,2,2,2,2];
    let c : &[isize] = &[2,2,2,2,3];
    let cc : &[isize] = &[2,2,2,2,2,2];

    println!("{:?}", a);

    assert!(a < b);
    assert!(a <= b);
    assert!(a != b);
    assert!(b >= a);
    assert!(b > a);

    println!("{:?}", b);

    assert!(b < c);
    assert!(b <= c);
    assert!(b != c);
    assert!(c >= b);
    assert!(c > b);

    assert!(a < c);
    assert!(a <= c);
    assert!(a != c);
    assert!(c >= a);
    assert!(c > a);

    println!("{:?}", c);

    assert!(a < cc);
    assert!(a <= cc);
    assert!(a != cc);
    assert!(cc >= a);
    assert!(cc > a);

    println!("{:?}", cc);
}
