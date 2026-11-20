//@ run-pass
#![allow(unused_mut)]
fn foo<T>(y: Option<T>) {
    let mut x: isize;
    let mut rs: Vec<isize> = Vec::new();
    /* tests that x doesn't get put in the precondition for the
       entire if expression */

    if true {
    } else {
        match y {
          None::<T> => x = 17,
          _ => x = 42
        }
        rs.push(x);
    }
    return;
}

pub fn main() { println!("hello"); foo::<isize>(Some::<isize>(5)); }
