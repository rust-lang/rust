// Regression test for #13466

pub fn main() {
    // The expected arm type `Option<T>` has one type parameter, while
    // the actual arm `Result<T, E>` has two. typeck should not be
    // tricked into looking up a non-existing second type parameter.
    let _x: usize = match Some(1) {
        Ok(u) => u,
        //~^ ERROR mismatched types
        //~| NOTE_NONVIRAL expected enum `Option<{integer}>`
        //~| NOTE_NONVIRAL found enum `Result<_, _>`
        //~| NOTE_NONVIRAL expected `Option<{integer}>`, found `Result<_, _>`

        Err(e) => panic!(e)
        //~^ ERROR mismatched types
        //~| NOTE_NONVIRAL expected enum `Option<{integer}>`
        //~| NOTE_NONVIRAL found enum `Result<_, _>`
        //~| NOTE_NONVIRAL expected `Option<{integer}>`, found `Result<_, _>`
    };
}
