// run-rustfix
#![allow(unused)]
#![warn(clippy::all)]

fn main() {
    let v = Some(true);
    let s = [0, 1, 2, 3, 4];
    match v {
        Some(x) => (),
        y @ _ => (),
    }
    match v {
        Some(x) => (),
        y @ None => (), // no error
    }
    match s {
        [x, inside @ .., y] => (), // no error
        [..] => (),
    }

    let mut mutv = vec![1, 2, 3];

    // required "ref" left out in suggestion: #5271
    match mutv {
        ref mut x @ _ => {
            x.push(4);
            println!("vec: {:?}", x);
        },
        ref y if y == &vec![0] => (),
    }

    match mutv {
        ref x @ _ => println!("vec: {:?}", x),
        ref y if y == &vec![0] => (),
    }
}
