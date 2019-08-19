// run-pass
#![allow(non_camel_case_types)]


type an_int = isize;

fn cmp(x: Option<an_int>, y: Option<isize>) -> bool {
    x == y
}

pub fn main() {
    assert!(!cmp(Some(3), None));
    assert!(!cmp(Some(3), Some(4)));
    assert!(cmp(Some(3), Some(3)));
    assert!(cmp(None, None));
}
