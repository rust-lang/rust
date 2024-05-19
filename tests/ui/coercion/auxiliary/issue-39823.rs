#![crate_type="rlib"]

#[derive(Debug, PartialEq)]
pub struct RemoteC(pub u32);

#[derive(Debug, PartialEq)]
pub struct RemoteG<T>(pub T);
