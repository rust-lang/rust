// run-pass
#![allow(non_camel_case_types)]
#![allow(non_shorthand_field_patterns)]

enum t1 { a(isize), b(usize), }
struct T2 {x: t1, y: isize}
enum t3 { c(T2, usize), }

fn m(input: t3) -> isize {
    match input {
      t3::c(T2 {x: t1::a(m), ..}, _) => { return m; }
      t3::c(T2 {x: t1::b(m), y: y}, z) => { return ((m + z) as isize) + y; }
    }
}

pub fn main() {
    assert_eq!(m(t3::c(T2 {x: t1::a(10), y: 5}, 4)), 10);
    assert_eq!(m(t3::c(T2 {x: t1::b(10), y: 5}, 4)), 19);
}
