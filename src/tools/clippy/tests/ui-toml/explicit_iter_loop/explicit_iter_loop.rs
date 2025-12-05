#![warn(clippy::explicit_iter_loop)]

fn main() {
    let mut vec = vec![1, 2, 3];
    let rmvec = &mut vec;
    for _ in rmvec.iter() {}
    //~^ ERROR: it is more concise to loop over references to containers
    for _ in rmvec.iter_mut() {}
    //~^ ERROR: it is more concise to loop over references to containers
}
