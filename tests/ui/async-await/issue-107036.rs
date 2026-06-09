//@ aux-build:issue-107036.rs
//@ edition:2021
//@ check-pass

extern crate issue_107036;
use issue_107036::S;

async fn f() {
    S{}.f().await;
}

fn main() {
    let _ = f();
}
