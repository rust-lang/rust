// error-pattern:attrs on macros are not yet supported

// Don't know how to deal with a syntax extension appearing after an
// item attribute. Probably could use a better error message.
#[foo = "bar"]
fmt!("baz")
fn main() { }