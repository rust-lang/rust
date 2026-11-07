//@ check-pass

#![expect(clippy::never_loop)]

async fn f() {
    loop {
        break;
    }
}

fn main() {}
