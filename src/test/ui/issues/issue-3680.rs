fn main() {
    match None {
        Err(_) => ()
        //~^ ERROR mismatched types
        //~| expected enum `std::option::Option<_>`
        //~| found enum `std::result::Result<_, _>`
        //~| expected enum `std::option::Option`, found enum `std::result::Result`
    }
}
