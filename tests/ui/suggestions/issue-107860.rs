//@ edition: 2021

async fn str<T>(T: &str) -> &str { &str }
//~^ ERROR mismatched types

fn main() {}
