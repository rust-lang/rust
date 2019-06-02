// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash
#![allow(dead_code)]
#![allow(unused_variables)]

use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::convert::TryFrom;

#[derive(Copy, Clone, Debug)]
struct MyBool(bool);

impl PartialEq<bool> for MyBool {
    fn eq(&self, other: &bool) -> bool {
        self.0 == *other
    }
}

impl PartialEq<MyBool> for bool {
    fn eq(&self, other: &MyBool) -> bool {
        *self == other.0
    }
}

fn check_boolean_array_with_const_generics<const LEN: usize>() {
    let a = [true; LEN];
    let mut b = [true; LEN];
    let mut c = [false; LEN];
    let mut d = [false; LEN];

    // PartialEq, Debug
    assert_eq!(a, b);
    if LEN > 0 {
        assert_ne!(a, c);
    } else {
        assert_eq!(a, c);
    }

    // AsRef
    let a_slice: &[bool] = a.as_ref();

    // AsMut
    let b_slice: &mut [bool] = b.as_mut();

    // Borrow
    let c_slice: &[bool] = c.borrow();

    // BorrowMut
    let d_slice: &mut [bool] = d.borrow_mut();

    // TryFrom
    let e_ref: &[bool; LEN] = TryFrom::try_from(a_slice).unwrap();

    // TryFrom#2
    let f: [bool; LEN] = TryFrom::try_from(a_slice).unwrap();

    // TryFrom#3
    let g_mut: &mut [bool; LEN] = TryFrom::try_from(b_slice).unwrap();

    // PartialEq, Eq, Hash
    let h: HashSet<[bool; LEN]> = HashSet::new();

    // PartialEq, Eq, PartialOrd, Ord
    let i: BTreeSet<[bool; LEN]> = BTreeSet::new();

    // IntoIterator#1
    for j in &a {
        let _ = j;
    }

    // IntoIterator#2
    for k in &mut c {
        let _ = k;
    }

    let l = [MyBool(true); LEN];
    let l_slice: &[MyBool] = l.as_ref();

    let mut m = [MyBool(false); LEN];
    let m_slice: &mut [MyBool] = m.as_mut();

    // PartialEq
    assert_eq!(a, l);
    assert_eq!(a, *l_slice);
    assert_eq!(a_slice, l);
    assert_eq!(a, l_slice);
    assert_eq!(c, m_slice);
    assert_eq!(m_slice, c);

    // The currently omitted impls
    /*
    assert_eq!(a, &l);
    assert_eq!(&l, a);
    assert_eq!(a, &mut l);
    assert_eq!(&mut l, a);
    */

    /* Default is not using const generics now */
    /*
    assert_eq!(c, Default::default())
    */
}

fn main() {
    check_boolean_array_with_const_generics::<{ 30 }>();
    check_boolean_array_with_const_generics::<{ 40 }>();
}
