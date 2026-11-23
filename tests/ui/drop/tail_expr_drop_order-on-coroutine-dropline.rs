//@ edition: 2021
//@ check-pass
//@ compile-flags: -W rust-2024-compatibility

pub struct S(String);

async fn conn(_: &String) -> Result<S, String> {
    Ok(S(String::new()))
}

pub async fn g(p: String) -> S {
    conn(&p).await.expect("x")
}

fn main() {}
