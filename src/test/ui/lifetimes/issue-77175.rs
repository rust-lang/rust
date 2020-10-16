#[deny(single_use_lifetimes)]
// edition:2018
// check-pass

// error: lifetime parameter `'a` only used once
async fn bar<'a>(s1: String, s2: &'_ str, s3: &'a str) -> &'a str {
    s3
}

fn foo<'a>(s1: String, s2: &'_ str, s3: &'a str) -> &'a str {
    s3
}

fn main() {}
