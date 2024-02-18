//@ check-pass
// a compared case for `import-after-macro-expand-6.rs`

pub mod a {
    pub use crate::b::*;
}

mod b {
    mod http {
        pub struct HeaderMap;
    }

    pub use self::http::*;
    pub struct HeaderMap;
}

use crate::a::HeaderMap;

fn main() {
    let h: crate::b::HeaderMap = HeaderMap;
}
