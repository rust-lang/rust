// Testing the output when an invalid builtin attribute is passed as value
// to `doc(attribute = "...")`.

#![feature(rustdoc_internals)]

#[doc(attribute = "foo df")] //~ ERROR
const _: () = ();

#[doc(attribute = "fooyi")] //~ ERROR
const _: () = ();
