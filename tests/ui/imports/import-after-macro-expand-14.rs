//@ check-pass

use crate::a::HeaderMap;

mod b {
    pub mod http {
        #[derive(Clone)]
        pub struct HeaderMap;
    }

    pub use self::http::*;
    #[derive(Debug)]
    pub struct HeaderMap;
}

mod a {
    pub use crate::b::*;
}

fn main() {
    let h: crate::b::HeaderMap = HeaderMap;
}
