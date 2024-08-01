unsafe extern "C" {
    //~^ ERROR extern block cannot be declared unsafe
}

// We can't gate `unsafe extern` blocks themselves since they were previously
// allowed, but we should gate the `safe` soft keyword.
#[cfg(any())]
unsafe extern "C" {
    safe fn foo();
    //~^ ERROR `unsafe extern {}` blocks and `safe` keyword are experimental
}

fn main() {}
