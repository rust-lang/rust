// edition: 2021

#![feature(async_fn_in_trait)]

trait A {
    async fn e() {
        Ok(())
        //~^ ERROR mismatched types
    }
}

fn main() {}
