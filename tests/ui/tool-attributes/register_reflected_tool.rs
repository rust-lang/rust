//@ aux-crate: dep=reflecting.rs
//@ compile-flags: -Zcrate-attr=register_tool(jjj(reflect))

#![feature(register_tool)]
#![feature(type_info)]

#![feature(const_trait_impl)]
#![feature(const_cmp)]

#![register_tool(aaa(reflect))]
#![register_tool(bbb)]
#![register_tool(ccc(reflect::c))] //~ ERROR unrecognized tool option `reflect::c`
#![register_tool(ddd(mrow))] //~ ERROR unrecognized tool option `mrow`
#![register_tool(eee(reflect, reflect))] //~ ERROR malformed `register_tool` attribute input
#![register_tool(fff(mrow, reflect))] //~ ERROR malformed `register_tool` attribute input

#![register_tool(ggg)]
#![register_tool(ggg(reflect))] //~ ERROR duplicate tool `ggg` registered

#![register_tool(hhh(reflect))]
#![register_tool(hhh)] //~ ERROR duplicate tool `hhh` registered



#![register_tool(clippy(reflect), rustfmt(reflect))]
//~^ ERROR the tool attribute `clippy` is predefined and cannot be marked as being available for reflection
//~| ERROR the tool attribute `rustfmt` is predefined and cannot be marked as being available for reflection
#![register_tool(diagnostic(reflect), miri(reflect), rust_analyzer(reflect))]
//~^ ERROR the tool attribute `diagnostic` is predefined and cannot be marked as being available for reflection
//~| ERROR the tool attribute `miri` is predefined and cannot be marked as being available for reflection
//~| ERROR the tool attribute `rust_analyzer` is predefined and cannot be marked as being available for reflection

use std::any::TypeId;

#[aaa::aa]
#[bbb::bb]
#[hhh::hh]
#[ggg::gg]
#[jjj::jj]
#[clippy::a]
#[rustfmt::a]
#[diagnostic::a]
//~^ WARN unknown diagnostic attribute
#[miri::a]
#[rust_analyzer::a]
struct A;

const _: () = assert!( TypeId::of::<A>().reflected_attrs() ==
    ["#[aaa::aa]", "#[hhh::hh]", "#[ggg::gg]", "#[jjj::jj]"] );

// test cross crate encoding of reflectivity
const _: () = assert!( TypeId::of::<dep::A>().reflected_attrs() == &["#[bbb::bb]"]);


fn main() {}
