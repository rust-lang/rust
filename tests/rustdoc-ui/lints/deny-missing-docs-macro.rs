//! foo

#![deny(missing_docs)]

#[macro_export]
macro_rules! foo { //~ ERROR
    () => {}
}
