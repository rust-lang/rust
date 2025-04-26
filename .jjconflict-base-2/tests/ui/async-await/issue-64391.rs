// Regression test for Issue #64391. The goal here is that this
// function compiles. In the past, due to incorrect drop order for
// temporaries in the tail expression, we failed to compile this
// example. The drop order itself is directly tested in
// `drop-order/drop-order-for-temporary-in-tail-return-expr.rs`.
//
//@ check-pass
//@ edition:2018

async fn add(x: u32, y: u32) -> u32 {
    async { x + y }.await
}

fn main() { }
