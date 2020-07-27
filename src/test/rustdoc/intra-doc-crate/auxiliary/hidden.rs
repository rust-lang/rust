#![crate_name = "hidden_dep"]
#![deny(intra_doc_link_resolution_failure)]

#[doc(hidden)]
pub mod __reexport {
    pub use crate::*;
}

pub mod future {
    mod ready {

        /// Link to [`ready`](function@ready)
        pub struct Ready;
        pub fn ready() {}

    }
    pub use self::ready::{ready, Ready};

}
