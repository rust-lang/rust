// https://github.com/rust-lang/rust/issues/123887
// Do not ICE on unsized extern parameter
//@ compile-flags: -Clink-dead-code --emit=link
#![feature(extern_types, unsized_fn_params)]

extern "C" {
    type ExternType;
}

struct Wrapper<T: ?Sized>(T);
fn f(_: Wrapper<ExternType>) {} //~ ERROR the size for values of type `ExternType` cannot be known at compilation time

fn main() {}
