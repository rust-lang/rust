#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::clippy::let_underscore_drop)]
#![allow(clippy::missing_docs_in_private_items)]

fn main() {
    let _: Vec<_> = vec![5; 6].into_iter().filter(|&x| x == 0).map(|x| x * 2).collect();

    let _: Vec<_> = vec![5_i8; 6]
        .into_iter()
        .filter(|&x| x == 0)
        .flat_map(|x| x.checked_mul(2))
        .collect();

    let _: Vec<_> = vec![5_i8; 6]
        .into_iter()
        .filter_map(|x| x.checked_mul(2))
        .flat_map(|x| x.checked_mul(2))
        .collect();

    let _: Vec<_> = vec![5_i8; 6]
        .into_iter()
        .filter_map(|x| x.checked_mul(2))
        .map(|x| x.checked_mul(2))
        .collect();
}
