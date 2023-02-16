// edition: 2021
// known-bug: #108142

#![feature(async_fn_in_trait)]

trait A {
    async fn e() {
        Ok(())
    }
}

fn main() {}
