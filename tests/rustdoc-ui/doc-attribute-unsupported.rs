// This is currently not supported but should be!

#![feature(rustdoc_internals)]

#[doc(attribute = "diagnostic::do_not_recommend")] //~ ERROR
/// bla
mod yup {}
