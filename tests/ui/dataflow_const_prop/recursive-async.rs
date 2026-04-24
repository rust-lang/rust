//! Ensure DataflowConstProp doesn't cause an error with async recursion as in #155376.

//@ edition:2018
//@ check-pass
//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+DataflowConstProp --crate-type=lib

pub async fn foo(n: usize) {
    if n > 0 {
        Box::pin(foo(n - 1)).await;
    }
}
