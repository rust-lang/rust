#![feature(rustdoc_internals)]
#![feature(register_tool)]
#![register_tool(custom_tool)]

//@ check-pass
#[doc(attribute = "custom_tool::foo")]
/// bla
const _: () = ();
