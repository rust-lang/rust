// run-pass
#![allow(unused_mut)]


pub fn main() {
    let mut later: Vec<isize> ;
    if true { later = vec![1]; } else { later = vec![2]; }
    println!("{}", later[0]);
}
