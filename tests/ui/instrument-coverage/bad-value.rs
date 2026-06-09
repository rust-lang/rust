//@ revisions: blank bad
//@ [blank] compile-flags: -Cinstrument-coverage=
//@ [bad] compile-flags: -Cinstrument-coverage=bad-value

fn main() {}

//[blank]~? ERROR incorrect value `` for codegen option `instrument-coverage`
//[bad]~? ERROR incorrect value `bad-value` for codegen option `instrument-coverage`
