// compile-flags: -Ztrait-solver=next
// known-bug: unknown
// failure-status: 101
// dont-check-compiler-stderr

// This test will fail until we fix `FulfillmentCtxt::relationships`. That's
// because we create a type variable for closure upvar types, which is not
// constrained until after we try to do fallback on diverging type variables.
// Thus, we will call that function, which is unimplemented.

fn require_fn(_: impl Fn() -> i32) {}

fn main() {
    require_fn(|| -> i32 { 1i32 });
}
