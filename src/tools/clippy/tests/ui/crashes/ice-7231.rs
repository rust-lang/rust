//@ check-pass

#![allow(clippy::never_loop)]

async fn f() {
    loop {
        break;
    }
}

fn main() {}
