//@ known-bug: #154296
//@ edition: 2024
mod m1 {
    mod inner {
        pub struct S;
    }
    pub use inner::*;
    #[derive(Debug)]
    pub struct S;
}
use m1::*;
use S;
