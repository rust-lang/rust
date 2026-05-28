//@ run-rustfix
fn main() {
    let v = vec![1, 2];
    let sum = v //~ ERROR: type annotations needed
        .iter()
        .map(|val| *val)
        .sum(); // `sum::<T>` needs `T` to be specified
    // In this case any integer would fit, but we resolve to `i32` because that's what `{integer}`
    // got coerced to. If the user needs further hinting that they can change the integer type, that
    // can come from other suggestions. (#100802)
    let bool = sum > 0;
    assert_eq!(bool, true);
}
