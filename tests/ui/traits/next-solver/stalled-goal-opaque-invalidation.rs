//@ check-pass
//@ edition: 2021
//@ compile-flags: -Znext-solver

// A recursive non-defining use of the async fn opaque relies on the
// registered opaque type to guide inference for the recursive result.
async fn mirror<T>(t: T) -> T {
    let value = Box::pin(mirror(String::new())).await;
    let _ = value.len();

    t
}

fn main() {}
