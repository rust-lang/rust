// Test for issue #139314
// This test ensures that the compiler doesn't panic with
// "called `Option::unwrap()` on a `None` value" when processing
// async functions with const parameters.

//@ edition:2018
//@ error-pattern: the trait bound

// This is a simplified version of the test case that caused the ICE
async fn func<T: Iterator<Item = u8> + Copy, const N: usize>(iter: T) -> impl for<'a1> Clone {
    func(iter.map(|x| x + 1))
}

fn main() {
    // Just make sure the function compiles, we don't need to call it
}
