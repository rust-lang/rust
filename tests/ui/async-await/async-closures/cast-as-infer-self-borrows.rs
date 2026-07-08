//@ edition:2021

// Regression test for #155999.

fn needs_fn_mut<T>(x: impl FnMut() -> T) {
    needs_fn_mut(async || x as _)
    //~^ ERROR type annotations needed
}

fn main() {}
