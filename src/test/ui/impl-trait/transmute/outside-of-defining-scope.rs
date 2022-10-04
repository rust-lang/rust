// For this to compile we have to reveal opaque types during typeck
// without using `Reveal::All` as that would cause a cycle when revealing `Cycle`.
//
// check-pass
#![feature(type_alias_impl_trait)]
#![crate_type = "lib"]
use std::mem::transmute;
fn foo() -> impl Sized {
    0u8
}

type Cycle = impl Sized;
fn define() -> Cycle {}

unsafe fn with_opaque_in_env() -> u8
where
    Cycle: Sized,
{
    transmute::<_, u8>(foo())
}
