// run-pass
#![allow(non_camel_case_types)]


struct boxed_int<'a> {
    f: &'a isize,
}

fn max<'r>(bi: &'r boxed_int, f: &'r isize) -> isize {
    if *bi.f > *f {*bi.f} else {*f}
}

fn with(bi: &boxed_int) -> isize {
    let i = 22;
    max(bi, &i)
}

pub fn main() {
    let g = 21;
    let foo = boxed_int { f: &g };
    assert_eq!(with(&foo), 22);
}
