#[crate_id = "foo#0.1"];

//! Very docs

pub mod bar {

    /// So correct
    pub mod baz {
        /// Much detail
        pub fn baz() { }
    }

    /// *wow*
    pub trait Doge { }
}
