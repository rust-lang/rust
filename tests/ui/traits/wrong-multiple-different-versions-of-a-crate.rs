// https://github.com/rust-lang/rust/issues/148892
//@ aux-crate:crate1=crate1.rs

struct MyStruct; //~ HELP  the trait `Trait` is not implemented for `MyStruct`

fn main() {
    crate1::foo(MyStruct); //~ ERROR the trait bound `MyStruct: Trait` is not satisfied
}
