fn main() {
    // Below we call the closure with its own return as the argument, unifying
    // its inferred input and return types. We want to make sure that the generated
    // error handles this gracefully, and in particular doesn't generate an extra
    // note about the `?` operator in the closure body, which isn't relevant to
    // the inference.
    let x = |r| {
        //~^ ERROR type annotations needed
        let v = r?;
        Ok(v)
    };

    let _ = x(x(Ok(())));
}
