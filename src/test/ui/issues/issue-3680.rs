fn main() {
    match None {
        Err(_) => ()
        //~^ ERROR mismatched types
        //~| expected enum `Option<_>`
        //~| found enum `Result<_, _>`
        //~| expected enum `Option`, found enum `Result`
    }
}
