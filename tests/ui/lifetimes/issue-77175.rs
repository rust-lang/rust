#[deny(single_use_lifetimes)]
// edition:2018
// check-pass

// Prior to the fix, the compiler complained that the 'a lifetime was only used
// once. This was obviously wrong since the lifetime is used twice: For the s3
// parameter and the return type. The issue was caused by the compiler
// desugaring the async function into a generator that uses only a single
// lifetime, which then the validator complained about becauase of the
// single_use_lifetimes constraints.
async fn bar<'a>(s1: String, s2: &'_ str, s3: &'a str) -> &'a str {
    s3
}

fn foo<'a>(s1: String, s2: &'_ str, s3: &'a str) -> &'a str {
    s3
}

fn main() {}
