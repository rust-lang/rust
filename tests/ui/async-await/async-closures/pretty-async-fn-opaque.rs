//@ edition: 2021

fn produce() -> impl AsyncFnMut() -> &'static str {
    async || ""
}

fn main() {
    let x: i32 = produce();
    //~^ ERROR mismatched types
}
