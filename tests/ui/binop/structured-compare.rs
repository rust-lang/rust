//@ run-pass

#![allow(non_camel_case_types)]


#[derive(Copy, Clone, Debug)]
enum foo { large, small, }

impl PartialEq for foo {
    fn eq(&self, other: &foo) -> bool {
        ((*self) as usize) == ((*other) as usize)
    }
    fn ne(&self, other: &foo) -> bool { !(*self).eq(other) }
}

pub fn main() {
    let a = (1, 2, 3);
    let b = (1, 2, 3);
    assert_eq!(a, b);
    assert!(a != (1, 2, 4));
    assert!(a < (1, 2, 4));
    assert!(a <= (1, 2, 4));
    assert!((1, 2, 4) > a);
    assert!((1, 2, 4) >= a);
    let x = foo::large;
    let y = foo::small;
    assert!(x != y);
    assert_eq!(x, foo::large);
    assert!(x != foo::small);
}
