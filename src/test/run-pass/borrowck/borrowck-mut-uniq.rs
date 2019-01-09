// run-pass
#![feature(box_syntax)]

use std::mem::swap;

#[derive(Debug)]
struct Ints {sum: Box<isize>, values: Vec<isize> }

fn add_int(x: &mut Ints, v: isize) {
    *x.sum += v;
    let mut values = Vec::new();
    swap(&mut values, &mut x.values);
    values.push(v);
    swap(&mut values, &mut x.values);
}

fn iter_ints<F>(x: &Ints, mut f: F) -> bool where F: FnMut(&isize) -> bool {
    let l = x.values.len();
    (0..l).all(|i| f(&x.values[i]))
}

pub fn main() {
    let mut ints: Box<_> = box Ints {sum: box 0, values: Vec::new()};
    add_int(&mut *ints, 22);
    add_int(&mut *ints, 44);

    iter_ints(&*ints, |i| {
        println!("isize = {:?}", *i);
        true
    });

    println!("ints={:?}", ints);
}
