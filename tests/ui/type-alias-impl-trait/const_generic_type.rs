//@edition: 2021
//@revisions: infer no_infer

#![feature(type_alias_impl_trait)]
type Bar = impl std::fmt::Display;

async fn test<const N: crate::Bar>() {
    //[no_infer]~^ ERROR: type annotations needed
    //~^^ ERROR: `Bar` is forbidden as the type of a const generic parameter
    #[cfg(infer)]
    let x: u32 = N;
}

fn main() {}
