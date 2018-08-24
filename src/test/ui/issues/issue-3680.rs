fn main() {
    match None {
        Err(_) => ()
        //~^ ERROR mismatched types
        //~| expected type `std::option::Option<_>`
        //~| found type `std::result::Result<_, _>`
        //~| expected enum `std::option::Option`, found enum `std::result::Result`
    }
}
