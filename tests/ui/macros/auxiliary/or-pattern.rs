#![crate_type = "lib"]

#[macro_export]
macro_rules! a {
    ($x:pat|) => ();
}
