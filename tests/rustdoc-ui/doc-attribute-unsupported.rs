// Invalid namespace attribute!

#![feature(rustdoc_internals)]

#[doc(attribute = "diagnostic::")] //~ ERROR
/// bla
const _: () = ();

#[doc(attribute = "foo::bar::foo")] //~ ERROR
/// bla
const _: () = ();

#[doc(attribute = "unknown_tool_attr::foo")] //~ ERROR
/// bla
const _: () = ();
