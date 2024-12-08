//@ check-pass
// similar as `import-after-macro-expand-6.rs`

use crate::a::HeaderMap;

mod b {
    pub mod http {
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
