// run-rustfix

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::missing_docs_in_private_items)]

fn main() {
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(|x| 0..x).flatten().collect();
    let _: Option<_> = (Some(Some(1))).map(|x| x).flatten();
}
