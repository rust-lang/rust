// edition: 2021
// ignore-compare-mode-lower-impl-trait-in-trait-to-assoc-ty

#![feature(async_fn_in_trait)]
//~^ WARN the feature `async_fn_in_trait` is incomplete and may not be safe to use and/or cause compiler crashes

trait A {
    async fn e() {
        Ok(())
        //~^ ERROR mismatched types
        //~| HELP consider using a semicolon here
    }
}

fn main() {}
