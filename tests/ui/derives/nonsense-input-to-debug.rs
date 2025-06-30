// Issue: #32950
// Ensure that using macros rather than a type doesn't break `derive`.

#[derive(Debug)]
struct Nonsense<T> {
    //~^ ERROR type parameter `T` is never used
    should_be_vec_t: vec![T],
    //~^ ERROR `derive` cannot be used on items with type macros
    //~| ERROR expected type, found `expr` metavariable
}

fn main() {}
