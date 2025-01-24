#![crate_name = "time"]
#![crate_type = "lib"]

// This code compiled without error in Rust 1.79, but started failing in 1.80
// after the addition of several `impl FromIterator<_> for Box<str>`.

pub fn parse() -> Option<Vec<()>> {
    let iter = std::iter::once(Some(())).map(|o| o.map(Into::into));
    let items = iter.collect::<Option<Box<_>>>()?; //~ ERROR E0282
    //~^ NOTE this is an inference error on crate `time` caused by an API change in Rust 1.80.0; update `time` to version `>=0.3.35`
    Some(items.into())
    //~^ NOTE type must be known at this point
}
