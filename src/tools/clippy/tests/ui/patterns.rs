//@aux-build:proc_macros.rs
#![allow(clippy::uninlined_format_args, clippy::single_match)]

#[macro_use]
extern crate proc_macros;

fn main() {
    let v = Some(true);
    let s = [0, 1, 2, 3, 4];
    match v {
        Some(x) => (),
        y @ _ => (),
        //~^ redundant_pattern
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
            //~^ redundant_pattern
            x.push(4);
            println!("vec: {:?}", x);
        },
        ref y if y == &vec![0] => (),
    }

    match mutv {
        ref x @ _ => println!("vec: {:?}", x),
        //~^ redundant_pattern
        ref y if y == &vec![0] => (),
    }
    external! {
        let v = Some(true);
        match v {
            Some(x) => (),
            y @ _ => (),
        }
    }
}
