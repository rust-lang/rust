// Regression test for #13466

pub fn main() {
    // The expected arm type `Option<T>` has one type parameter, while
    // the actual arm `Result<T, E>` has two. typeck should not be
    // tricked into looking up a non-existing second type parameter.
    let _x: usize = match Some(1) {
        Ok(u) => u,
        //~^ ERROR mismatched types
        //~| expected type `std::option::Option<{integer}>`
        //~| found type `std::result::Result<_, _>`
        //~| expected enum `std::option::Option`, found enum `std::result::Result`

        Err(e) => panic!(e)
        //~^ ERROR mismatched types
        //~| expected type `std::option::Option<{integer}>`
        //~| found type `std::result::Result<_, _>`
        //~| expected enum `std::option::Option`, found enum `std::result::Result`
    };
}
