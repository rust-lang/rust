//@ run-pass
#![allow(non_camel_case_types)]




// Tests for if as expressions returning nominal types

#[derive(Copy, Clone)]
struct I { i: isize }

fn test_rec() {
    let rs = if true { I {i: 100} } else { I {i: 101} };
    assert_eq!(rs.i, 100);
}

#[derive(Copy, Clone, Debug)]
enum mood { happy, sad, }

impl PartialEq for mood {
    fn eq(&self, other: &mood) -> bool {
        ((*self) as usize) == ((*other) as usize)
    }
    fn ne(&self, other: &mood) -> bool { !(*self).eq(other) }
}

fn test_tag() {
    let rs = if true { mood::happy } else { mood::sad };
    assert_eq!(rs, mood::happy);
}

pub fn main() { test_rec(); test_tag(); }
