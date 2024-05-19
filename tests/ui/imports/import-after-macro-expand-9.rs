//@ check-pass

use crate::b::*;

mod b {
    pub mod http {
        pub struct HeaderMap;
    }

    pub use self::http::*;
    #[derive(Debug)]
    pub struct HeaderMap;
}

fn main() {
    let h: crate::b::HeaderMap = HeaderMap;
}
