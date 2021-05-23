// edition:2018
#![allow(clippy::never_loop)]

async fn f() {
    loop {
        break;
    }
}

fn main() {}
