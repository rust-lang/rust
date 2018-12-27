// run-pass
#![allow(dead_code)]

enum BogusOption<T> {
    None,
    Some(T),
}

type Iterator = isize;

pub fn main() {
    let x = [ 3, 3, 3 ];
    for i in &x {
        assert_eq!(*i, 3);
    }
}
