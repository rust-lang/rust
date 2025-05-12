// Regression test for #54239, shouldn't trigger lint.
//@ check-pass
//@ edition:2018

#![deny(missing_debug_implementations)]

struct DontLookAtMe(i32);

async fn secret() -> DontLookAtMe {
    DontLookAtMe(41)
}

pub async fn looking() -> i32 { // Shouldn't trigger lint here.
    secret().await.0
}

fn main() {}
