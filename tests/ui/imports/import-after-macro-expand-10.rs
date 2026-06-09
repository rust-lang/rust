//@ check-pass

mod b {
    pub mod http {
        pub struct HeaderMap;
    }

    pub use self::http::*;
    #[derive(Debug)]
    pub struct HeaderMap;
}

use crate::b::*;

fn main() {
    let h: crate::b::HeaderMap = HeaderMap;
}
