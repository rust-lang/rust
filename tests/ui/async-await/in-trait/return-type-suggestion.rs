// edition: 2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]

trait A {
    async fn e() {
        Ok(())
        //~^ ERROR mismatched types
        //~| HELP consider using a semicolon here
    }
}

fn main() {}
