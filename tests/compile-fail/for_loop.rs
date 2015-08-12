#![feature(plugin)]
#![plugin(clippy)]

#[deny(needless_range_loop)]
fn main() {
    let vec = vec![1, 2, 3, 4];
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
}
