//@ edition: 2021
#![allow(incomplete_features)]
#![feature(move_expr)]

use std::sync::Arc;

fn main() {
    let c = Arc::new(String::new());
    let _future = async {
        let f = async {
            drop(move(c.clone()));
        };
        f.await;
        drop(c);
    };
    println!("{c}"); //~ ERROR the type `Arc` does not implement `Copy`
}
