//@ run-pass
#![allow(unused_assignments)]
#![allow(unused_variables)]
#![allow(non_shorthand_field_patterns)]

pub fn main() {
    struct Foo { x: isize, y: isize }
    let mut f = Foo { x: 10, y: 0 };
    match f {
        Foo { ref mut x, .. } => *x = 11,
    }
    match f {
        Foo { ref x, ref y } => {
            assert_eq!(f.x, 11);
            assert_eq!(f.y, 0);
        }
    }
    match f {
        Foo { mut x, y: ref mut y } => {
            x = 12;
            *y = 1;
        }
    }
    assert_eq!(f.x, 11);
    assert_eq!(f.y, 1);
}
