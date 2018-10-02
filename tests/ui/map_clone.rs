#![feature(tool_lints)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_docs_in_private_items)]

fn main() {
    let _: Vec<i8> = vec![5_i8; 6].iter().map(|x| *x).collect();
    let _: Vec<String> = vec![String::new()].iter().map(|x| x.clone()).collect();
    let _: Vec<u32> = vec![42, 43].iter().map(|&x| x).collect();
}
