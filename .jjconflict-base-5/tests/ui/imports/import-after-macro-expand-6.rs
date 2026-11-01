//@ check-pass
// https://github.com/rust-lang/rust/pull/113099#issuecomment-1633574396

pub mod a {
    pub use crate::b::*;
}

mod b {
    pub mod http {
        pub struct HeaderMap;
    }

    pub use self::http::*;
    #[derive(Debug)]
    pub struct HeaderMap;
}

use crate::a::HeaderMap;

fn main() {
    let h: crate::b::HeaderMap = HeaderMap;
}
