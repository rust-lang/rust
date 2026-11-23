//@ aux-crate: dep=const-eval-doc-attrs.rs
//@ run-pass

#![feature(type_info, const_cmp, const_trait_impl)]
use std::any::TypeId;


// should be encoded since UI test is of CrateType::Executable
///AAA
#[doc = "BBB"]
struct A;

const _: () = assert!( dep::N == 0 ) ;
const _: () = const { assert!( TypeId::of::<A>().docs() == &["AAA", "BBB"] ) };
const _: () = const { assert!( TypeId::of::<dep::C>().docs() == &["CCC", "DDD"] ) };



fn main() {
}
