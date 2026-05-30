//! Test for invalid MetaItem syntax in the attribute

#![crate_type = "lib"]
#![feature(rustc_attrs)]

#[rustc_on_unimplemented( //~ ERROR malformed `rustc_on_unimplemented` attribute input [E0539]
    message="the message"
    label="the label"
)]
trait T {}
