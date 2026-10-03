//! Test type inference in method chains with diverging fallback.
//! Verifies that closure type in `unwrap_or_else` is properly inferred
//! when chained with other combinators and contains a diverging path.

//@ run-pass

fn produce<T>() -> Result<&'static str, T> {
    Ok("22")
}

fn main() {
    // The closure's error type `T` must unify with `ParseIntError`,
    // while the success type must be `usize` (from parse())
    let x: usize = produce()
        .and_then(|x| x.parse::<usize>()) // Explicit turbofish for clarity
        .unwrap_or_else(|_| panic!()); // Diverging fallback

    assert_eq!(x, 22);
}
