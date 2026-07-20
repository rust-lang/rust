//@ edition: 2024
//@ check-pass

// Regression test for https://github.com/rust-lang/rust/issues/159427.
// This test keeps nested awaited futures below the query depth limit.
// Do not add a `recursion_limit` attribute here.
// The default limit is the test input.

macro_rules! nested_async {
    () => {
        async {}
    };
    ($_head:tt $($tail:tt)*) => {
        async { nested_async!($($tail)*).await }
    };
}

fn main() {
    let _ = nested_async!(
        x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x
    );
}
