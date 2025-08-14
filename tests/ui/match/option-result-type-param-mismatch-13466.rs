//! Regression test for https://github.com/rust-lang/rust/issues/13466

// Regression test for #13466

//@ dont-require-annotations: NOTE

pub fn main() {
    // The expected arm type `Option<T>` has one type parameter, while
    // the actual arm `Result<T, E>` has two. typeck should not be
    // tricked into looking up a non-existing second type parameter.
    let _x: usize = match Some(1) {
        Ok(u) => u,
        //~^ ERROR mismatched types
        //~| NOTE expected enum `Option<{integer}>`
        //~| NOTE found enum `Result<_, _>`
        //~| NOTE expected `Option<{integer}>`, found `Result<_, _>`

        Err(e) => panic!(e)
        //~^ ERROR mismatched types
        //~| NOTE expected enum `Option<{integer}>`
        //~| NOTE found enum `Result<_, _>`
        //~| NOTE expected `Option<{integer}>`, found `Result<_, _>`
    };
}
