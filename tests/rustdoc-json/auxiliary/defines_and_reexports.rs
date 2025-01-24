pub mod m1 {
    pub struct InPubMod;
}

mod m2 {
    pub struct InPrivMod;
}

pub use m1::{InPubMod, InPubMod as InPubMod2};
pub use m2::{InPrivMod, InPrivMod as InPrivMod2};
