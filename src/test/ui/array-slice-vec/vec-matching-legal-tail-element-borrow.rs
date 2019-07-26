// run-pass
#![allow(unused_variables)]

#![feature(slice_patterns)]

pub fn main() {
    let x = &[1, 2, 3, 4, 5];
    let x: &[isize] = &[1, 2, 3, 4, 5];
    if !x.is_empty() {
        let el = match x {
            &[1, ref tail..] => &tail[0],
            _ => unreachable!()
        };
        println!("{}", *el);
    }
}
