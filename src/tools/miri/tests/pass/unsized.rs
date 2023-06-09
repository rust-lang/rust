//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(unsized_tuple_coercion)]
#![feature(unsized_fn_params)]

use std::mem;

fn unsized_tuple() {
    let x: &(i32, i32, [i32]) = &(0, 1, [2, 3]);
    let y: &(i32, i32, [i32]) = &(0, 1, [2, 3, 4]);
    let mut a = [y, x];
    a.sort();
    assert_eq!(a, [x, y]);

    assert_eq!(&format!("{:?}", a), "[(0, 1, [2, 3]), (0, 1, [2, 3, 4])]");
    assert_eq!(mem::size_of_val(x), 16);
}

fn unsized_params() {
    pub fn f0(_f: dyn FnOnce()) {}
    pub fn f1(_s: str) {}
    pub fn f2(_x: i32, _y: [i32]) {}
    pub fn f3(_p: dyn Send) {}

    let c: Box<dyn FnOnce()> = Box::new(|| {});
    f0(*c);
    let foo = "foo".to_string().into_boxed_str();
    f1(*foo);
    let sl: Box<[i32]> = [0, 1, 2].to_vec().into_boxed_slice();
    f2(5, *sl);
    let p: Box<dyn Send> = Box::new((1, 2));
    f3(*p);
}

fn main() {
    unsized_tuple();
    unsized_params();
}
