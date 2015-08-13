#![feature(plugin)]
#![plugin(clippy)]

#[deny(needless_range_loop, explicit_iter_loop)]
fn main() {
    let mut vec = vec![1, 2, 3, 4];
    let vec2 = vec![1, 2, 3, 4];
    for i in 0..vec.len() {      //~ERROR the loop variable `i` is only used to index `vec`.
        println!("{}", vec[i]);
    }
    for i in 0..vec.len() {      //~ERROR the loop variable `i` is used to index `vec`.
        println!("{} {}", vec[i], i);
    }
    for i in 0..vec.len() {      // not an error, indexing more than one variable
        println!("{} {}", vec[i], vec2[i]);
    }

    for _v in vec.iter() { } //~ERROR it is more idiomatic to loop over `&vec`
    for _v in vec.iter_mut() { } //~ERROR it is more idiomatic to loop over `&mut vec`

    for _v in &vec { } // these are fine
    for _v in &mut vec { } // these are fine
}
