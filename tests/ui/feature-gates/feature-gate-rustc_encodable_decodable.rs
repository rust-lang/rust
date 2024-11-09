#![crate_type = "lib"]

// This isn't intended to compile, so it's easiest to just ignore this error.
extern crate rustc_serialize; //~ERROR can't find crate for `rustc_serialize`

#[derive(
    RustcEncodable,
    //~^   ERROR   use of unstable library feature `rustc_encodable_decodable`
    //~^^  WARNING this was previously accepted by the compiler
    //~^^^ WARNING use of deprecated macro `RustcEncodable`
    RustcDecodable,
    //~^   ERROR   use of unstable library feature `rustc_encodable_decodable`
    //~^^  WARNING this was previously accepted by the compiler
    //~^^^ WARNING use of deprecated macro `RustcDecodable`
)]
struct S;
