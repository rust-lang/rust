#![crate_name = "time"]
#![crate_type = "lib"]

//@check-pass

// This code compiled without error in Rust 1.79, but started failing in 1.80
// after the addition of several `impl FromIterator<_> for Box<str>`.

pub fn parse() -> Option<Vec<()>> {
    let iter = std::iter::once(Some(())).map(|o| o.map(Into::into));
    let items = iter.collect::<Option<Box<_>>>()?;
    Some(items.into())
}
