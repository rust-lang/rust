//@ known-bug: rust-lang/rust#126267

#![feature(transmutability)]
#![crate_type = "lib"]

pub enum ApiError {}
pub struct TokioError {
    b: bool,
}
pub enum Error {
    Api { source: ApiError },
    Ethereum,
    Tokio { source: TokioError },
}

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src>, // safety is NOT assumed
    {
    }
}

fn test() {
    struct Src;
    type Dst = Error;
    assert::is_transmutable::<Src, Dst>();
}
