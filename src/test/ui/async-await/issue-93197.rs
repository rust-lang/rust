// Regression test for #93197
// build-pass
// edition:2021

#![feature(try_blocks)]

use std::sync::{mpsc, mpsc::SendError};

pub async fn foo() {
    let (tx, _) = mpsc::channel();

    let _: Result<(), SendError<&str>> = try { tx.send("hello")?; };
}

fn main() {}
