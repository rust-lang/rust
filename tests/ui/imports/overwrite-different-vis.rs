//@ check-pass

mod b {
    pub mod http {
        pub struct HeaderMap;
    }

    pub(crate) use self::http::*;
    #[derive(Debug)]
    pub struct HeaderMap;
}

mod a {
    pub use crate::b::*;

    fn check_type() {
        let _: HeaderMap = crate::b::HeaderMap;
    }
}

fn main() {}
