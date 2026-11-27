// Capture a coherence pattern from wasm-bindgen that we discovered as part of
// future-compatibility warning #56105. This pattern currently receives a lint
// warning but we probably want to support it long term.
//
// Key distinction: we are implementing once for `A` (take ownership) and one
// for `&A` (borrow).
//
// c.f. #56105

#![deny(coherence_leak_check)]

trait IntoWasmAbi {
    fn some_method(&self) {}
}

trait FromWasmAbi {}
trait RefFromWasmAbi {}
trait ReturnWasmAbi {}

impl<'a, 'b, A, R> IntoWasmAbi for &'a (dyn Fn(A) -> R + 'b)
where
    A: FromWasmAbi,
    R: ReturnWasmAbi,
{
}

// Explicitly writing the bound lifetime.
impl<'a, 'b, A, R> IntoWasmAbi for &'a (dyn for<'x> Fn(&'x A) -> R + 'b)
where
    A: RefFromWasmAbi,
    R: ReturnWasmAbi,
{
    //~^^^^^ ERROR conflicting implementation
    //~| WARN the behavior may change in a future release
}

fn main() {}
