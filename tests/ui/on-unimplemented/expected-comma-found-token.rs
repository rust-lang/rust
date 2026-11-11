//! Test for invalid MetaItem syntax in the attribute

#![crate_type = "lib"]
#![feature(rustc_attrs)]

#[rustc_on_unimplemented(
    message="the message" //~ ERROR attribute items not separated with `,`
    label="the label"
)]
trait T {}
