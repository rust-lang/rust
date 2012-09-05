
use std;

fn foo<T>(y: Option<T>) {
    let mut x: int;
    let mut rs: ~[int] = ~[];
    /* tests that x doesn't get put in the precondition for the
       entire if expression */

    if true {
    } else {
        match y {
          None::<T> => x = 17,
          _ => x = 42
        }
        rs += ~[x];
    }
    return;
}

fn main() { debug!("hello"); foo::<int>(Some::<int>(5)); }
