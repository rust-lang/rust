fn main() {
    match None {
        Err(_) => ()
        //~^ ERROR mismatched types
        //~| expected enum `Option<_>`
        //~| found enum `std::result::Result<_, _>`
        //~| expected enum `Option`, found enum `std::result::Result`
    }
}
