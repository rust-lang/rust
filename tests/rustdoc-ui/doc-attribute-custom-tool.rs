#![feature(rustdoc_internals)]
#![feature(register_tool)]
#![register_tool(custom_tool)]

#[doc(attribute = "custom_tool::foo")]
/// bla
const _: () = ();
