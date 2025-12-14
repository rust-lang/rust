// https://github.com/rust-lang/rust/pull/113099#issuecomment-1633574396

pub mod a {
    pub use crate::b::*;
    //~^ WARN ambiguous glob re-exports
    //~| WARN ambiguous glob re-exports
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
//~^ ERROR `HeaderMap` is ambiguous
//~| WARN this was previously accepted

fn main() {
    let h: crate::b::HeaderMap = HeaderMap;
    //~^ ERROR `HeaderMap` is ambiguous
    //~| WARN this was previously accepted
}
